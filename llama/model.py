"""LLaMA baseline for the TinyBalls-V1 validation plan (run 1).

Plain decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU. No KDA, no
Engram, no MTP — those are for runs 2 and 3. Tied embeddings.

Precision layout (T4 = sm75, fp16 autocast, no bf16):
- body (norms, rope, linears, SDPA) runs in fp16 tensor cores with the
  residual stream kept in fp16 too — the fp32 promotion lives only inside
  RMSNorm's reduction and the CE upcast. This halves elementwise traffic
  and backward save sizes; rope tables are cached on device and pre-cast
  to the activation dtype so SDPA never sees mixed q/k/v dtypes (mixed
  dtypes would silently drop the fused memory-efficient kernel on sm75
  and fall back to the fp32 math path).
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

from llama.kda import KDA

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
    # run 2 hybrid pattern (layer indices, 0-based): KDA at 1-3, 5-7, 9-11
    # (1-indexed 2-4, 6-8, 10-12); full attention at 0, 4, 8
    kda_layers=(1, 2, 3, 5, 6, 7, 9, 10, 11),
)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # fp32 for the reduction regardless of autocast state; gain and
        # output go back out in x.dtype so the residual stream stays fp16
        # under autocast (fp32 here would double elementwise traffic and
        # backward save sizes for everything downstream)
        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.to(x.dtype) * self.weight.to(x.dtype)


def rope_cache(seq_len, head_dim, theta, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # (seq, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq, head_dim)
    return emb.cos().to(dtype)[None, None, :, :], emb.sin().to(dtype)[None, None, :, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


_SDPA_PROBED = False


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
        global _SDPA_PROBED
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
        if not _SDPA_PROBED:
            _SDPA_PROBED = True
            print(f"[attn probe] q/k/v dtype {q.dtype}, "
                  f"uniform={q.dtype == k.dtype == v.dtype}, head_dim {self.head_dim}",
                  flush=True)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(out)


SWIGLU_CLAMP = 10.0


class MLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        # clamp at 10 like the frontier stacks: free fp16 stability, silu
        # saturates well before 10 so the clamp only trims outliers
        return self.down_proj(F.silu(self.gate_proj(x).clamp(max=SWIGLU_CLAMP))
                              * self.up_proj(x).clamp(max=SWIGLU_CLAMP))


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


class FullAttention(nn.Module):
    """Run 2 full-attention layer: GQA + QK-norm + learned sink + output
    gate, NoPE. QK-norm (RMSNorm over head_dim on q and k) keeps logits in a
    tame range with no positional encoding; the sink is a per-head learned
    logit appended to the softmax (softmax over T+1 with a zero value), the
    standard learned-attention-sink of the 2026 stacks. Output gate mirrors
    the KDA layers so both sublayers gate their output the same way.

    Attention materializes the (T+1) score matrix — the sink can't ride on
    SDPA — so the sublayer is gradient-checkpointed by the block to keep
    activation memory at SDPA levels.
    """

    def __init__(self, dim, n_heads, n_kv_heads, head_dim, rms_eps):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        # per-head-dim QK-norm weights, shared across heads
        self.q_norm = RMSNorm(head_dim, rms_eps)
        self.k_norm = RMSNorm(head_dim, rms_eps)
        # learned sink logit per q head (init 0: e^0 = 1 unit of softmax mass)
        self.attn_sink = nn.Parameter(torch.zeros(n_heads))
        self.o_gate = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq, self.n_kv_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = q @ k.transpose(-1, -2) * self.head_dim ** -0.5  # (B, H, T, T)
        # sink column: constant logit per head, no value contribution
        sink = self.attn_sink.to(scores.dtype).view(1, -1, 1, 1)
        scores = torch.cat([scores, sink.expand(bsz, -1, seq, 1)], dim=-1)
        probs = F.softmax(scores, dim=-1)[..., :-1]  # drop sink column
        out = probs @ v
        out = out.transpose(1, 2).reshape(bsz, seq, -1)
        return self.o_proj(torch.sigmoid(self.o_gate(x)) * out)


class HybridBlock(nn.Module):
    """Block whose attention sublayer is either KDA or full attention, picked
    by layer index. Full-attention layers drop RoPE (NoPE): local order in the
    hybrid stack is carried by the KDA layers' short convs and recurrent
    state, mirroring the frontier hybrid designs where exact position
    encoding is confined to (or dropped from) the few full layers."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        if layer_idx in cfg["kda_layers"]:
            self.attn = KDA(cfg["dim"], n_heads=cfg["n_heads"],
                            head_dim=cfg["head_dim"])
            self.is_kda = True
        else:
            self.attn = FullAttention(cfg["dim"], cfg["n_heads"],
                                      cfg["n_kv_heads"], cfg["head_dim"],
                                      cfg["rms_eps"])
            self.is_kda = False
        self.mlp_norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        self.mlp = MLP(cfg["dim"], cfg["ffn_dim"])
        # Measured at micro 8 / seq 2048 (dim 768, 12x64): an uncheckpointed
        # KDA sublayer saves ~1.8GB of fp32 intermediates for backward —
        # q/k/v + conv in/out + all the fp32 kernel internals (ke_pos/ke_neg,
        # the Ag solve, per-chunk state) — essentially the same as an
        # uncheckpointed full-attention layer. Nine KDA layers at that rate
        # is ~16GB before weights/grads/Adam, which OOM'd a 16GB T4 at
        # initial eval. So checkpoint EVERY attention sublayer, both types:
        # backward re-runs the scan (~2x KDA fwd compute, acceptable on T4)
        # and the saved-tensor peak drops to MLP+norm sized. The KDA kernel's
        # internal autocast(enabled=False) holds inside cp.checkpoint's
        # recompute, so the fp32-contract fixes stay in force on the re-pass.
        self.checkpoint_attn = True

    def forward(self, x, cos, sin):
        h = self.attn_norm(x)
        if self.checkpoint_attn:
            import torch.utils.checkpoint as cp
            a = cp.checkpoint(self.attn, h, use_reentrant=False)
        else:
            a = self.attn(h)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x


class HybridModel(nn.Module):
    """Run 2: KDA 3:1 hybrid. Layers 1-3, 5-7, 9-11 (1-indexed) are KDA;
    4, 8, 12 are full attention (GQA + QK-norm + sink + gate, NoPE).
    Everything outside the attention pattern matches the LLaMA baseline so
    run 2 isolates the attention swap.

    Param note: the KDA gate/output paths add ~1M per KDA layer over a GQA
    layer, so this model lands at ~124M vs the baseline's 113M (+9.6%). The
    proposal's "active params within a few percent" tolerance is exceeded
    because Kimi-style KDA carries a low-rank gate path (f_a/f_b, g_a/g_b)
    plus per-channel gate projections that GQA simply doesn't have. Trimming
    the FFN or embedding to compensate would confound the comparison with a
    second variable; the cleaner control is the KDA_V_HEADS knob (grouped
    value attention, Kimi's own efficiency lever) which trades KDA params
    for compute directly. Recorded here so the delta is a known quantity in
    the writeup, not a surprise."""

    def __init__(self, **overrides):
        super().__init__()
        cfg = dict(DEFAULTS)
        cfg.update(overrides)
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["dim"])
        self.blocks = nn.ModuleList(HybridBlock(cfg, i)
                                    for i in range(cfg["n_layers"]))
        self.norm = RMSNorm(cfg["dim"], cfg["rms_eps"])
        self.lm_head = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self._rope = None
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
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, targets=None, ce_chunk=4096):
        bsz, seq = tokens.shape
        x = self.tok_emb(tokens)
        if x.device.type == "cuda":
            x = x.half()
        cos, sin = self._rope_tables(seq, tokens.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            if targets is None:
                return self.lm_head(x.float())
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
        seen = set()
        n = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            n += p.numel()
        return n


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
        if x.device.type == "cuda":
            x = x.half()  # fp16 residual stream; norms still reduce in fp32
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
