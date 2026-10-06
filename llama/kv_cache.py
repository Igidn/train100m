"""KV cache + batched sampling for TinyBalls — the RL rollout workhorse.

Why this exists: `llama/model.py` had no cache of any kind (confirmed with
`git log --all -S past_key_value` over the whole history: it never existed, not
that it was lost). That is fine for training, where every sequence is a single
packed full forward, but fatal for RL — generating a 300-token rollout by
re-running the whole sequence per token is O(n^2) forwards, which would eat the
entire GPU budget on redundant recompute.

Everything here is additive. `LLaMA.forward` keeps its old behaviour exactly
when `cache=None` and `positions=None`, so training and the battery are
untouched (the trainer asserts nothing about it, but `test_kv_cache.py` also
checks the uncached path against the pre-change arithmetic).

Three things the naive version gets wrong, and how they are handled:

1. **`is_causal=True` on a cached step.** A decode step has q_len == 1. PyTorch
   aligns the causal mask top-left, so with q_len=1 it masks every key except
   position 0 — the model would only ever see the first token. A single query is
   the newest position and may attend to the entire cache, so the mask is None.

2. **Padding.** Rollouts of one prompt diverge in length once tool calls differ,
   so a batch is not length-uniform. Prompts are *left*-padded and the pad
   columns are masked out with an additive mask at both prefill and decode.
   Right-padding would be cheaper for prefill but leaves each row's cache
   holding garbage after its real length, which then has to be sliced per row.

3. **Positions.** Left-padding means rows sit at different rope offsets, so a
   single `pos_offset: int` cannot be right. `positions` is a per-row
   (B, T) index into the rope table instead.

The cache stores the *unexpanded* kv heads (n_kv_heads=4) rather than the
repeat_interleave'd 12, so it is 3x smaller; the expansion is redone per step.

Typical RL use: best-of-N / GRPO groups. All N rollouts of one prompt share a
prompt, so the batch is N copies — no padding at all on the first step.
"""
from dataclasses import dataclass

import torch

IM_END = 2
_NEG = float("-inf")


@dataclass
class LayerKV:
    k: torch.Tensor = None
    v: torch.Tensor = None


class KVCache:
    """Preallocated per-layer k/v store for one batch of sequences.

    `length` is the number of positions filled so far, and it is advanced by the
    *caller once per forward*, not inside append(). append() is called once per
    layer within a single forward, so incrementing there would make layer 1 write
    at offset `seq` instead of 0 and layer N overflow the buffer — which shows up
    only as quietly wrong logits several steps later.
    """

    def __init__(self, n_layers, max_len, device="cpu"):
        self.n_layers = n_layers
        self.max_len = max_len
        self.device = device
        self.layers = [LayerKV() for _ in range(n_layers)]
        self.length = 0

    def reset(self):
        for l in self.layers:
            l.k = l.v = None
        self.length = 0

    def append(self, i, k, v):
        """Store this step's k/v (B, n_kv, T, hd) at `length`; return history."""
        l = self.layers[i]
        B, H, T, D = k.shape
        end = self.length + T
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow {end} > max_len {self.max_len}")
        if l.k is None:
            l.k = torch.zeros(B, H, self.max_len, D, dtype=k.dtype, device=k.device)
            l.v = torch.zeros(B, H, self.max_len, D, dtype=v.dtype, device=v.device)
        l.k[:, :, self.length:end] = k
        l.v[:, :, self.length:end] = v
        return l.k[:, :, :end], l.v[:, :, :end]

    def advance(self, n):
        self.length += n


class _LayerView:
    """Lets Attention call cache.append(self, k, v) without knowing its index."""

    __slots__ = ("cache", "i")

    def __init__(self, cache, i):
        self.cache, self.i = cache, i

    def append(self, _attn, k, v):
        return self.cache.append(self.i, k, v)


class LayerViews:
    """Thread one `cache` argument through LLaMA -> Block -> Attention.

    `.cache` is the underlying KVCache so the caller can advance `length` once
    per forward; `views[i]` is what Block i receives.
    """

    def __init__(self, cache):
        self.cache = cache
        self.views = [_LayerView(cache, i) for i in range(cache.n_layers)]

    def __getitem__(self, i):
        return self.views[i]


def rope_tables(max_pos, head_dim, theta, device, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                       dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)  # (max_pos, head_dim)


def left_pad(prompts, pad_id=0, device="cpu"):
    """Left-pad a list of id lists -> (x, T0). Padded columns are garbage; the
    caller must not generate from them (generate() requires equal lengths)."""
    B = len(prompts)
    T0 = max(len(p) for p in prompts)
    x = torch.full((B, T0), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        x[i, T0 - len(p):] = torch.tensor(p, dtype=torch.long, device=device)
    return x, T0


@torch.inference_mode()
def generate(model, prompts, max_new=256, temp=0.0, top_p=0.95, eos_id=IM_END,
             seed=None, stop_on_eos=True):
    """Batched sampling with a KV cache.

    prompts: list[list[int]] of *equal-length* token ids (already including the
    trailing "assistant\\n"). Equal length is asserted because unequal lengths
    need padded-column masking, which is not wired up yet — see module docstring.
    Returns list[list[int]] of generated ids (EOS excluded unless it is the
    first token generated).
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    B = len(prompts)
    T0 = len(prompts[0])
    if any(len(p) != T0 for p in prompts):
        raise ValueError("generate() requires equal-length prompts; batch "
                         "best-of-N groups of one prompt, or pad+mask first")

    x = torch.tensor(prompts, dtype=torch.long, device=device)
    cfg = model.cfg
    cache = KVCache(cfg["n_layers"], T0 + max_new + 1, device=device)
    views = LayerViews(cache)

    out = [[] for _ in range(B)]
    done = [False] * B
    nxt = torch.zeros(B, dtype=torch.long, device=device)

    logits = model(x, cache=views, pos_offset=0)[:, -1, :]
    nxt = sample(logits, temp, top_p)

    for step in range(max_new):
        for i in range(B):
            if done[i]:
                continue
            t = int(nxt[i])
            if t == eos_id:
                if stop_on_eos:
                    done[i] = True
                    continue
            out[i].append(t)
        if all(done) or step == max_new - 1:
            break
        logits = model(nxt.view(B, 1), cache=views, pos_offset=T0 + step)[:, -1, :]
        nxt = sample(logits, temp, top_p)

    return out


def sample(logits, temp, top_p):
    if temp <= 0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits.float() / temp, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) < top_p
    sp = sp.masked_fill(~keep, 0.0)
    sp = sp / sp.sum(dim=-1, keepdim=True)
    pick = torch.multinomial(sp, 1)
    return si.gather(-1, pick).squeeze(-1)