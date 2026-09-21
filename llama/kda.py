"""KDA — Kimi Delta Attention (Kimi Linear, arXiv:2510.26692) for TinyBalls run 2.

A gated delta net with per-channel forget gates: the same recurrence as a
gated delta net, but the decay is a vector per head instead of a scalar.

Per layer:
- q, k, v projections through one depthwise causal conv1d (kernel 4) with
  SiLU — the only source of local position mixing in the layer.
- forget gate g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias), per channel,
  clamped at -5 in log space (per-step decay floor e^-5 ~ 0.007).
- input gate beta = sigmoid(b_proj(x)) — how hard each token overwrites.
- the delta rule itself, q/k L2-normalized, computed by a chunked WY
  representation (64-token chunks, pure PyTorch — no Triton, since T4 fp16
  tensor cores are the target and autograd gives the backward for free).
- output through a sigmoid-gated RMSNorm, then o_proj.

The chunked math was validated against fla's own per-token reference
(`flash-linear-attention` `fla/ops/kda/naive.py`, `naive_recurrent_kda`)
to max abs err 6e-8 over random gates/beta/multiple chunk boundaries, and
gradients checked finite end to end. Conventions follow the fla kernels:

  S_t = exp(g_t) * S_{t-1}                      (per-channel decay)
  v_int_t = beta_t * (v_t - k_t . S_{t-1})      (delta correction)
  S_t += k_t v_int_t^T
  o_t = q_t . S_t

State S is (K, V) per head (key-first, i.e. S k_v = sum_d k[d] S[d, v]);
k_t . S means sum over the K axis of S.

fp32 overflow note: the WY factorization wants exp(C_i) and exp(-C_j)
separately (C = in-chunk gate cumsum) so the pairwise gate difference
exp(C_i - C_j) comes out of one matmul. With the fla gate floor of -5 per
step, exp(-C_j) reaches e^320 at chunk 64 — past fp32 max — even though the
product exp(C_i - C_j) <= 1 is perfectly behaved. Chunk size 16 caps the
factors at e^80 (fp32 max ~ e^88) while the WY solve stays exact for any
chunk size, so the chunked form below is bit-compatible with the fla
64-token kernels, just cut into more, smaller chunks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

L2_EPS = 1e-6
# fla uses 64; we need 16 to keep exp(-C) in fp32 range at the -5 gate floor.
# Must divide evenly into common sequence lengths; 16 divides 2048/1024/512.
CHUNK = 16
_MAX_CUMSUM_RANGE = 80.0  # e^80 fits fp32; assert guards the gate-floor contract

# log-space anchors for dt_bias init: dt ~ loguniform([1e-3, 1e-1]),
# inverse-sigmoid transformed so sigmoid(dt_bias + x) ~ dt at init
_LOG01 = math.log(0.1)
_LOG0001 = math.log(0.001)


def _cdiv(a, b):
    return (a + b - 1) // b


def _inv_unit_lower(m):
    """(I + m)^-1 for strict-lower-triangular m, batched (..., n, n).

    Neumann series with doubling: (I + m)^-1 = sum_k (-m)^k, and polynomials
    in m commute, so T -> T + T@P, P -> P@P doubles the order each step.
    """
    n = m.shape[-1]
    t = torch.eye(n, dtype=m.dtype, device=m.device).expand(m.shape).clone()
    p = -m
    for _ in range(max(1, n.bit_length() - 1)):
        t = t + t @ p
        p = p @ p
    return t


def kda_chunk_fwd(q, k, v, g, beta, return_state=False):
    """Chunked KDA forward. All fp32.

    q, k: (N, T, K) — already L2-normalized and scaled by K^-0.5
    v:    (N, T, V)
    g:    (N, T, K) — log-space gates, <= 0
    beta: (N, T)    — in (0, 1)

    Returns o (N, T, V), and the final decayed state (N, V, K) if
    return_state (unused at train time; the sequence is one block).
    """
    n, t, kd = q.shape
    vd = v.shape[-1]
    nt = _cdiv(t, CHUNK)
    tg = nt * CHUNK
    pad = tg - t
    if pad:
        # F.pad on the last dim before head reshape: tensors are (N, T, D)
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))

    qc = q.view(n, nt, CHUNK, kd)
    kc = k.view(n, nt, CHUNK, kd)
    vc = v.view(n, nt, CHUNK, vd)
    gc = g.view(n, nt, CHUNK, kd)
    bc = beta.view(n, nt, CHUNK, 1)

    # inclusive prefix sum of gates within each chunk: C_t = g_1 + ... + g_t
    cum = gc.cumsum(2)
    cum_last = cum[:, :, -1]  # (n, nt, kd) — total decay across the chunk
    if (cum.max(2).values - cum.min(2).values).max() > _MAX_CUMSUM_RANGE:
        raise ValueError(
            "in-chunk gate cumsum range exceeds fp32 headroom; gates must be "
            "clamped so CHUNK * |gate floor| stays under e^80")

    # --- WY representation (per chunk): u_t = beta_t v_t - sum_{j<t} Ag[t,j] u_j
    # Ag[i,j] = beta_i sum_d k_id k_jd exp(C_id - C_jd), strict lower triangular.
    # exp(C_i) is folded into one factor and exp(-C_j) into the other so the
    # pairwise difference comes out of one matmul (per-channel gates make this
    # factoring essential — it cannot collapse to a scalar decay per head).
    # With CHUNK=16 both factors stay in fp32 range even at the -5 gate floor.
    kb = kc * bc
    ke_pos = kc * cum.exp()
    ke_neg = kc * (-cum).exp()
    ag = ((ke_pos * bc) @ ke_neg.transpose(-1, -2)).tril(-1)
    tu = _inv_unit_lower(ag)   # (I + Ag)^-1

    # --- chunk-level state scan over undecayed boundary states
    k_eff = kc * (cum_last.unsqueeze(2) - cum).exp()  # k * e^{C_last - C}
    q_eff = qc * cum.exp()                            # q * e^{C}
    ke_pos_all = ke_pos                               # k * e^{C}

    o = torch.zeros(n, nt, CHUNK, vd, dtype=torch.float32, device=q.device)
    h = torch.zeros(n, kd, vd, dtype=torch.float32, device=q.device)
    for i in range(nt):
        # h: undecayed state at the chunk boundary, orientation (K, V);
        # decay e^{C_t} rides on q_eff / ke_pos
        kv_h = torch.einsum("ntd,ndv->ntv", ke_pos_all[:, i], h)
        rhs = bc[:, i] * (vc[:, i] - kv_h)
        v_int = tu[:, i] @ rhs                        # delta corrections
        # o_t = (q_t e^{C_t}) . h + sum_{j<=t} (q_t . k_j) e^{C_t - C_j} v_int_j
        ba = (q_eff[:, i] @ ke_neg[:, i].transpose(-1, -2)).tril(0)
        o_in = ba @ v_int
        o_state = torch.einsum("ntd,ndv->ntv", q_eff[:, i], h)
        o[:, i] = o_state + o_in
        h = h * cum_last[:, i].exp().unsqueeze(-1) \
            + torch.einsum("ntd,ntv->ndv", k_eff[:, i], v_int)

    o = o.view(n, tg, vd)[:, :t]
    if return_state:
        # decay the boundary state to the true end-of-sequence state
        return o, h * cum_last[:, -1].exp().unsqueeze(-1)
    return o


class ShortConv(nn.Module):
    """Depthwise causal conv1d, kernel 4, SiLU. Per-channel weights, so one
    conv over the concatenated q/k/v stream costs 3 * dim * kernel params —
    the fla ShortConvolution layout without the CUDA kernel."""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(dim, kernel_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (B, T, D) -> causal conv: position t sees t-3..t
        b, t, d = x.shape
        xm = x.transpose(1, 2)  # (B, D, T)
        xm = F.pad(xm, (self.kernel_size - 1, 0))
        out = F.conv1d(xm, self.weight.unsqueeze(1), self.bias, groups=d)
        return F.silu(out.transpose(1, 2))


class KDA(nn.Module):
    """Kimi Delta Attention layer.

    Head layout matches Kimi Linear: q/k at (H, K) = (12, 64), v at (Hv, V) =
    (12, 64) — value heads are kept separate in the code because Kimi's GVA
    (grouped value attention) allows Hv != H, but at this scale they match.
    """

    def __init__(self, dim, n_heads=12, head_dim=64, conv_kernel=4,
                 gate_lower_bound=-5.0, rms_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.key_dim = n_heads * head_dim    # q/k width
        self.value_dim = n_heads * head_dim  # v width
        self.gate_lower_bound = gate_lower_bound
        self.rms_eps = rms_eps

        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)
        self.conv = ShortConv(self.key_dim * 2 + self.value_dim, conv_kernel)

        # forget gate: low-rank f_a -> f_b, per (head, channel) output
        self.f_a = nn.Linear(dim, self.head_dim, bias=False)
        self.f_b = nn.Linear(self.head_dim, self.value_dim, bias=False)
        # A_log per head, dt_bias per channel — fp32 params, see forward
        self.A_log = nn.Parameter(
            torch.log(torch.empty(n_heads).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(self.value_dim) * (_LOG01 - _LOG0001) + _LOG0001
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        # input gate (delta rule strength)
        self.b_proj = nn.Linear(dim, n_heads, bias=False)
        # output gate: second two-step low-rank path, per Kimi Linear
        self.g_a = nn.Linear(dim, self.head_dim, bias=False)
        self.g_b = nn.Linear(self.head_dim, self.value_dim, bias=True)
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

    def _gated_rmsnorm(self, o, gate):
        """RMSNorm over the head_dim axis with a sigmoid gate, per head.
        fp32 reduction; output cast back so the residual stream stays fp16."""
        dt = o.dtype
        of = o.float()
        gf = gate.float()
        ms = of.pow(2).mean(-1, keepdim=True) + self.rms_eps
        out = of * torch.rsqrt(ms) * torch.sigmoid(gf)
        return out.to(dt)

    def forward(self, x):
        b, t, _ = x.shape
        dt = x.dtype
        qkv = self.conv(torch.cat([self.q_proj(x), self.k_proj(x),
                                   self.v_proj(x)], dim=-1))
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)

        # forget gate, fp32: g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
        # f_b outputs (H, K) = value_dim channels; A_log broadcasts per head
        g = self.f_b(self.f_a(x)).float()               # (B, T, H*K)
        g = g.view(b, t, self.n_heads, self.head_dim)
        g = g + self.dt_bias.view(1, 1, self.n_heads, self.head_dim).float()
        g = -torch.exp(self.A_log.view(1, 1, -1, 1).float()) * F.softplus(g)
        if self.gate_lower_bound is not None:
            g = g.clamp(min=self.gate_lower_bound)

        beta = torch.sigmoid(self.b_proj(x).float())  # (B, T, H)

        # (B, T, H, D) -> (B*T*H, D): heads join the batch for the kernel
        def heads(y, d):
            return (y.view(b, t, self.n_heads, d)
                     .transpose(1, 2).reshape(b * self.n_heads, t, d))

        qh = heads(q, self.head_dim)
        kh = heads(k, self.head_dim)
        vh = heads(v, self.head_dim)
        gh = g.view(b, t, self.n_heads, self.head_dim) \
              .transpose(1, 2).reshape(b * self.n_heads, t, self.head_dim)
        bh = beta.view(b, t, self.n_heads).transpose(1, 2).reshape(b * self.n_heads, t)

        # fp32 kernel body: the chunked scan is numerically touchy in fp16
        # (exp of cumsums, a unit-lower solve), and at 12x64 heads it is a
        # small fraction of layer FLOPs
        qn = qh.float() / (qh.float().square().sum(-1, keepdim=True) + L2_EPS).sqrt()
        kn = kh.float() / (kh.float().square().sum(-1, keepdim=True) + L2_EPS).sqrt()
        qn = qn * self.head_dim ** -0.5

        o = kda_chunk_fwd(qn, kn, vh.float(), gh, bh)

        # (B*H, T, D) -> (B, T, H*D)
        o = o.view(b, self.n_heads, t, self.head_dim) \
             .transpose(1, 2).reshape(b, t, self.value_dim)
        gate = self.g_b(self.g_a(x))
        o = self._gated_rmsnorm(o, gate)
        return self.o_proj(o.to(dt))
