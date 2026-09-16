"""LLaMA baseline for the TinyBalls-V1 validation plan (run 1).

Plain decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU. No KDA, no
Engram, no MTP — those are for runs 2 and 3. Tied embeddings.

Precision layout (T4 = sm75, fp16 autocast, no bf16):
- body (norms, rope, linears, SDPA) runs in fp16 tensor cores; rope tables
  are cached on device and pre-cast to the activation dtype so SDPA never
  sees mixed q/k/v dtypes (mixed dtypes would silently drop the fused
  memory-efficient kernel on sm75 and fall back to the fp32 math path).
- LM head matmuls run in fp16 tensor cores; logits are quantized to fp16
  then softmax/CE runs on the fp32 upcast per chunk, and chunk losses are
  summed in fp32 — the fp16-saturating part of the tied-49k-vocab head
  (softmax over inf logits) still sees fp32 values. A logit past fp16
  range becomes inf before the upcast, which the train-loop skip path
  handles by backing off the loss scale.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(
    vocab_size=49154,
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=4,
    head_dim=64,
    ffn_dim=2048,
    rope_theta=10000.0,
    rms_eps=1e-5,
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # fp32 for the reduction regardless of autocast state
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out.to(x.dtype)) * self.weight


def rope_cache(seq_len, head_dim, theta, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, rope_theta, rms_eps):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(self, x, cos, sin):
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # cos/sin arrive pre-cast to x.dtype: fp16 math all the way, so q/k/v
        # stay fp16 and the fused memory-efficient kernel stays usable on sm75
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        # expand kv heads manually: enable_gqa pushes sm75 SDPA onto the math
        # backend, which materializes and retains the full fp32 attention matrix
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        self.attn = Attention(cfg["dim"], cfg["n_heads"], cfg["n_kv_heads"],
                              cfg["head_dim"], cfg["rope_theta"], cfg["rms_eps"])
        self.mlp_norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        self.mlp = MLP(cfg["dim"], cfg["ffn_dim"])

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LLaMA(nn.Module):
    def __init__(self, **overrides):
        super().__init__()
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg["n_layers"]))
        self.norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        self.lm_head = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self._rope = None  # cached (cos, sin) tables
        self._rope_key = None
        self.apply(self._init_weights)

    def _rope_tables(self, seq, device, dtype):
        key = (seq, device, dtype)
        if self._rope_key != key:
            self._rope = rope_cache(seq, self.cfg["head_dim"], self.cfg["rope_theta"],
                                    device, torch.float32)
            self._rope = (self._rope[0].to(dtype), self._rope[1].to(dtype))
            self._rope_key = key
        return self._rope

    def _init_weights(self, m):
        # LLaMA-style: small normal everywhere; nn.Embedding's default std=1
        # is fatal for a tied head (initial loss was ~ln(V) e+600 instead of ln(V))
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, targets=None, ce_chunk=4096):
        """tokens: (B, T) int64. Returns (loss,) if targets given, else logits."""
        bsz, seq = tokens.shape
        x = self.tok_emb(tokens)
        cos, sin = self._rope_tables(seq, tokens.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if targets is None:
                return self.lm_head(x.float())
            # chunked fp16 head: one weight cast per forward, tensor-core
            # gemms, fp32 softmax via cross_entropy on the fp32 upcast, and
            # fp32 summation of the per-chunk losses
            h = x.half() if x.device.type == "cuda" else x
            w = self.lm_head.weight.to(h.dtype)
            total = torch.zeros((), device=h.device, dtype=torch.float32)
            flat_h = h.reshape(-1, h.shape[-1])
            flat_t = targets.reshape(-1)
            for i in range(0, flat_h.shape[0], ce_chunk):
                logits = F.linear(flat_h[i:i + ce_chunk], w)
                total = total + F.cross_entropy(logits, flat_t[i:i + ce_chunk],
                                                reduction="none").float().sum()
            return total / flat_t.numel()

    def num_params(self):
        # tied embedding counted once
        seen = set()
        n = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n += p.numel()
        return n
