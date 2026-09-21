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
  representation. Two interchangeable backends produce the same math:

  * torch — pure PyTorch, 16-token chunks. The state scan is a python loop
    over T/16 chunks; numerically contract-bound (see below) and
    launch-overhead-heavy on T4. Default only when fla is unavailable.
  * fla  — the flash-linear-attention Triton kernel (`fla.ops.kda.chunk_kda`,
    64-token chunks, fp32 gate handling inside the kernel, fp16 tensor-core
    dots). fla explicitly supports pre-Ampere cards (it pins
    TRITON_F32_DEFAULT=ieee below sm80), and its chunk-level recompute
    replaces the python-level sublayer checkpoint. ~1 min of one-time Triton
    compile per session. Selected at startup by resolve_kda_impl(), which
    runs both backends on a small case and falls back to torch on any
    mismatch — so a broken fla install can never silently change the run.

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

AUTOCAST: the torch kernel is fp32 by contract and callers run under fp16
autocast (accelerate on T4), which downcasts matmul inputs even when they are
already fp32. kda_chunk_fwd therefore wraps itself in autocast(enabled=False).
Do not inline or reorder these matmuls without keeping that guard: ke_neg
design-reaches e^25..e^80 and fp16 max is ~e^11, so one autocast matmul turns
Ag into inf/NaN (this exact bug NaN'd run 2 at initial eval). The fla backend
needs no such guard — its kernels manage precision internally.

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
    """Chunked KDA forward, pure-PyTorch backend. All fp32.

    q, k: (N, T, K) — already L2-normalized and scaled by K^-0.5
    v:    (N, T, V)
    g:    (N, T, K) — log-space gates, <= 0
    beta: (N, T)    — in (0, 1)

    Returns o (N, T, V), and the final decayed state (N, V, K) if
    return_state (unused at train time; the sequence is one block).
    """
    n, t, kd = q.shape
    vd = v.shape[-1]
    # fp32 kernel, for real this time: callers run under fp16 autocast
    # (accelerate on T4), and autocast downcasts matmul INPUTS even when they
    # are already fp32. ke_neg reaches e^25..e^80 by design and e^11 is the
    # fp16 max, so an autocast matmul here turns Ag into inf/NaN. Nothing in
    # this function may run under autocast.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _kda_chunk_fwd(q, k, v, g, beta, return_state)


def _kda_chunk_fwd(q, k, v, g, beta, return_state=False):
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
    # The gate floor (-5) * CHUNK (16) bounds the in-chunk cumsum range at
    # exactly e^80; that bound is what keeps exp(-C) inside fp32, so it is a
    # construction invariant, not a runtime condition. It was once asserted
    # per call, but the comparison forces a GPU->CPU sync per layer per
    # micro and the clamp above already guarantees it — see train-loop
    # comment on syncs.

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

    # transposed once, outside the loop: bmm on the transposed batch view
    # is the same GEMM the einsum produced, minus per-iteration string parse
    # and permute overhead
    ke_neg_t = ke_neg.transpose(-1, -2)
    k_eff_t = k_eff.transpose(-1, -2)
    dec = cum_last.exp()  # per-chunk boundary decay, hoisted out of the loop

    o = torch.zeros(n, nt, CHUNK, vd, dtype=torch.float32, device=q.device)
    h = torch.zeros(n, kd, vd, dtype=torch.float32, device=q.device)
    for i in range(nt):
        # h: undecayed state at the chunk boundary, orientation (K, V);
        # decay e^{C_t} rides on q_eff / ke_pos
        kv_h = torch.bmm(ke_pos[:, i], h)
        rhs = bc[:, i] * (vc[:, i] - kv_h)
        v_int = torch.bmm(tu[:, i], rhs)                        # delta corrections
        # o_t = (q_t e^{C_t}) . h + sum_{j<=t} (q_t . k_j) e^{C_t - C_j} v_int_j
        ba = torch.bmm(q_eff[:, i], ke_neg_t[:, i]).tril(0)
        o[:, i] = torch.bmm(q_eff[:, i], h) + torch.bmm(ba, v_int)
        h = h * dec[:, i].unsqueeze(-1) + torch.bmm(k_eff_t[:, i], v_int)

    o = o.view(n, tg, vd)[:, :t]
    if return_state:
        # decay the boundary state to the true end-of-sequence state
        return o, h * dec[:, -1].unsqueeze(-1)
    return o


# ---------------------------------------------------------------- fla path

_FLA_CHUNK_KDA = None
_FLA_BROKEN = False


def _fla_chunk_kda(q, k, v, g, beta, scale):
    """flash-linear-attention Triton backend. Same math as kda_chunk_fwd,
    64-token chunks with the gate exp handled per-element inside the kernel
    (never factored into exp(+C)/exp(-C), so no fp32 overflow at the -5 floor
    and no python loop over chunks).

    Call contract (fla >= 0.5): q/k/v [B, T, H, D] (may be fp16 under
    autocast — the kernel upcasts for its internal math), L2 normalization of
    q/k done inside the kernel (use_qk_l2norm_in_kernel), g log-space
    pre-computed [B, T, H, K], beta post-sigmoid [B, T, H].
    """
    global _FLA_CHUNK_KDA, _FLA_BROKEN
    if _FLA_BROKEN:
        raise RuntimeError("fla backend previously failed; use the torch path")
    if _FLA_CHUNK_KDA is None:
        from fla.ops.kda import chunk_kda
        _FLA_CHUNK_KDA = chunk_kda
    o, _ = _FLA_CHUNK_KDA(
        q, k, v, g, beta,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
    return o


def resolve_kda_impl(requested, device):
    """Pick the KDA scan backend. "fla" wins only if it imports, runs, and
    matches the validated torch path on a small case; anything else falls
    back to torch, so a broken fla install can never poison a run."""
    if requested == "torch":
        return "torch"
    if requested not in ("fla", "auto"):
        raise ValueError(f"unknown KDA_IMPL {requested!r} (torch|fla|auto)")
    if device.type != "cuda":
        if requested == "fla":
            print("[kda] fla needs CUDA; cpu smoke uses the torch scan", flush=True)
        return "torch"
    try:
        from fla.ops.kda import chunk_kda  # noqa: F401
    except Exception as e:
        print(f"[kda] fla import failed ({e}); using the pure-PyTorch scan "
              f"(KDA_IMPL=auto fell back)", flush=True)
        return "torch"
    b, t, h, d, seed = 2, 128, 12, 64, 1234
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(b, t, h, d, generator=gen).to(device)
    k = torch.randn(b, t, h, d, generator=gen).to(device)
    v = torch.randn(b, t, h, d, generator=gen).to(device)
    g = (-torch.rand(b, t, h, d, generator=gen) * 3).clamp(min=-5.0).to(device)
    beta = torch.rand(b, t, h, generator=gen).to(device)
    with torch.no_grad():
        # torch reference: normalize outside (same contract as KDA.forward),
        # fla normalizes in-kernel — both fp32 math on the same inputs
        def heads(y):
            return y.transpose(1, 2).reshape(b * h, t, d)
        beta_h = beta.transpose(1, 2).reshape(b * h, t)
        qn = heads(q).float() / (heads(q).float().square().sum(-1, keepdim=True) + L2_EPS).sqrt()
        kn = heads(k).float() / (heads(k).float().square().sum(-1, keepdim=True) + L2_EPS).sqrt()
        ref = kda_chunk_fwd(qn * d ** -0.5, kn, heads(v).float(), heads(g), beta_h)
        got = None
        try:
            got = _fla_chunk_kda(q, k, v, g, beta, d ** -0.5)
        except Exception as e:
            print(f"[kda] fla chunk_kda failed ({type(e).__name__}: {e}); "
                  f"using the pure-PyTorch scan", flush=True)
            return "torch"
        got = got.transpose(1, 2).reshape(b * h, t, d).float()
        err = (got - ref).abs().max().item()
        if not torch.isfinite(got).all() or err > 2e-3:
            print(f"[kda] fla backend mismatch (max abs err {err:.2e}); "
                  f"using the pure-PyTorch scan", flush=True)
            return "torch"
        print(f"[kda] fla backend validated (max abs err {err:.2e} vs torch scan)",
              flush=True)
        return "fla"


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

    impl: "torch" (pure-PyTorch fp32 chunked scan) or "fla" (Triton kernel
    from flash-linear-attention; same math, resolved/validated at startup by
    resolve_kda_impl — never constructed with "fla" unless that passed).
    """

    def __init__(self, dim, n_heads=12, head_dim=64, conv_kernel=4,
                 gate_lower_bound=-5.0, rms_eps=1e-5, impl="torch"):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.key_dim = n_heads * head_dim    # q/k width
        self.value_dim = n_heads * head_dim  # v width
        self.gate_lower_bound = gate_lower_bound
        self.rms_eps = rms_eps
        self.impl = impl

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
        fp32 reduction; output cast back so the residual stream stays fp16.
        Autocast off: sigmoid on the fp32 gate would be downcast to fp16 and
        the exp-family ops in here share the kernel's overflow sensitivity."""
        dt = o.dtype
        with torch.autocast(device_type=o.device.type, enabled=False):
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
        # f_b outputs (H, K) = value_dim channels; A_log broadcasts per head.
        # Autocast off: the low-rank gate linears run fp32 by contract, and
        # softplus/exp on autocast fp16 would round the gates toward 0,
        # corrupting the decay the scan's fp32-range invariant rests on
        # (gate floor -5 x CHUNK 16 = e^80).
        xf = x.float()  # one fp32 copy shared by both gate paths
        with torch.autocast(device_type=x.device.type, enabled=False):
            g = self.f_b(self.f_a(xf)).float()                # (B, T, H*K)
            g = g.view(b, t, self.n_heads, self.head_dim)
            g = g + self.dt_bias.view(1, 1, self.n_heads, self.head_dim).float()
            g = -torch.exp(self.A_log.view(1, 1, -1, 1).float()) * F.softplus(g)
            if self.gate_lower_bound is not None:
                g = g.clamp(min=self.gate_lower_bound)

            beta = torch.sigmoid(self.b_proj(xf).float())     # (B, T, H)

        if self.impl == "fla":
            # fla kernels take (B, T, H, D) and normalize q/k in fp32 inside;
            # the fp16 activations stay in fp16 storage — no (B*H, T, D)
            # copies, no python chunk loop, no fp32 materialization of the
            # scan intermediates (the kernel recomputes in backward itself,
            # which is why the hybrid runs with KDA sublayers uncheckpointed)
            q4 = q.reshape(b, t, self.n_heads, self.head_dim)
            k4 = k.reshape(b, t, self.n_heads, self.head_dim)
            v4 = v.reshape(b, t, self.n_heads, self.head_dim)
            o4 = _fla_chunk_kda(q4, k4, v4, g, beta, self.head_dim ** -0.5)
            o = o4.reshape(b, t, self.value_dim)
        else:
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
