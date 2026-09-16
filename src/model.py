"""LLaMA baseline for the TinyBalls-V1 validation plan (run 1).

Plain decoder-only transformer: RMSNorm, RoPE, GQA, SwiGLU. No KDA, no
Engram, no MTP — those are for runs 2 and 3. Tied embeddings.
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
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
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

    def forward(self, tokens, targets=None, ce_chunk=8192):
        """tokens: (B, T) int64. Returns (loss,) if targets given, else logits."""
        bsz, seq = tokens.shape
        cos, sin = rope_cache(seq, self.cfg["head_dim"], self.cfg["rope_theta"],
                              tokens.device, torch.float32)
        x = self.tok_emb(tokens)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        if targets is None:
            return self.lm_head(x)
        # chunked cross-entropy: keeps the (B*T, vocab) logits tensor small
        total = x.new_zeros((), dtype=torch.float32)
        flat_h = x.reshape(-1, x.shape[-1])
        flat_t = targets.reshape(-1)
        for i in range(0, flat_h.shape[0], ce_chunk):
            logits = self.lm_head(flat_h[i:i + ce_chunk])
            total = total + F.cross_entropy(logits.float(), flat_t[i:i + ce_chunk],
                                            reduction="sum")
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
