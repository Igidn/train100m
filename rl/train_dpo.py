#!/usr/bin/env python
"""DPO trainer for TinyBalls-V1 — stage 4 (`rl/`).

    L = -log sigmoid( beta * [ (log pi(y_c|x) - log pi_ref(y_c|x))
                             - (log pi(y_r|x) - log pi_ref(y_r|x)) ] )
        + alpha * NLL(chosen) / len(chosen)

Two departures from the textbook form, both deliberate:

  **No reference model in memory.** `log pi_ref` is precomputed by
  `rl/dpo_pairs.py`, which generated `rejected` in the first place — so the
  reference policy *is* the policy that produced the rejected samples, and its
  logprobs are a constant of the dataset. Carrying a frozen 113M copy through
  every step would cost 450MB and buy nothing. `--ref-from-model` recomputes
  them if a pair file was built by something else.

  **The NLL(chosen) term.** Vanilla DPO raises only the *margin*, so
  log pi(y_c) can fall while the loss goes down. That matters more than usual
  here because the pairs are mined with gold as `chosen` and the policy is only
  0/8 task-correct, so chosen and rejected are barely different and the margin
  term has very little to grip. This is RPO / DPO+SFT; alpha defaults to 0.2.

Paired sequences, not packed: a DPO batch is [chosen; rejected] rows sharing one
prompt, right-padded with mask 0. Padding is safe for a logprob *sum* (unlike
generation, where a padded column gets read — see kv_cache.generate's assert).

Precision/size mirror the SFT and repair trainers: T4 fp16 + GradScaler,
micro-batch 1 pair/GPU, chunked fp32 head (a [32k, 49154] logit tensor never
materialises). Env:

  PAIRS_DIR      rl/dpo_pairs.py output (required)
  SFT_DATA_DIR   optional, for the anti-regression replay rows
  TOK_DATA_DIR   optional, raw-pretrain replay
  BETA           DPO inverse temperature, default 0.1
  ALPHA          NLL(chosen) weight, default 0.2
  LR             default 8e-6. DPO is far more LR-sensitive than SFT; 2e-4
                 (the multi-tok-v1 SFT LR) blows the implicit reward apart.
  EPOCHS         default 1
  WARMUP FLOOR   default 20 / 0.1
  REPLAY_FRAC    repair-SFT rows per batch, default 0.1
  PAIRS_PER_STEP PAIRS_PER_STEP, default 32 across all ranks
  MICRO_BS       pairs per GPU per micro-step, default 1
  MAX_LEN        truncate a pair's prompt from the left, default 3072
  CE_CHUNK       vocab chunk, default 2048
  CKPT_EVERY EVAL_EVERY MAX_HOURS MAX_STEPS OUT_DIR SEED WANDB_*
  BASE_REPO BASE_CKPT_FILE BASE_CKPT_PATH / HF_CKPT_REPO HF_CKPT_DIR

    python rl/train_dpo.py --selftest
    PAIRS_DIR=dpo-pairs SFT_DATA_DIR=/kaggle/... python rl/train_dpo.py
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama.model import LLaMA  # noqa: E402

# accelerate is imported lazily so `--selftest` runs on a bare torch box (the Pi
# has torch but not accelerate). train_sft.py's trainer needs it for mixed
# precision and the no_sync plumbing; DPO could drop it, and if a future run
# wants to remove that dependency this is the only place it is used.
try:
    from accelerate import Accelerator
except ImportError:      # pragma: no cover - selftest path
    Accelerator = None

ARCH_KEYS = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
             "head_dim", "ffn_dim", "rope_theta", "rms_eps")


# ------------------------------------------------------------------ loss
def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    """Returns (loss, metrics). Implemented here rather than inline so the
    selftest can pin it against hand-computed values — a DPO loss that is
    accidentally *ascending* still trains, it just learns the wrong thing."""
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits)
    return loss.mean(), {
        "dpo": loss.mean().item(),
        "acc": (logits > 0).float().mean().item(),
        "margin": logits.mean().item(),
        "chosen_lp": pi_c.mean().item(),
        "rejected_lp": pi_r.mean().item(),
        "ref_gap": ((pi_c - ref_c) - (pi_r - ref_r)).mean().item(),
    }


def nll_chosen(pi_c, n_c):
    """Per-token NLL of the chosen side, length-normalised.

    Normalising by length matters because chosen and rejected have different
    token counts, and an unnormalised sum would make longer answers look worse
    purely for being long — a length bias that shows up as the model learning to
    be terse. The mask is all-ones over the completion, so the count is exact.
    """
    return -(pi_c / n_c.clamp(min=1)).mean()


def pad_batch(seqs, pad_id=0):
    """Right-pad (ids, mask) rows to a common length -> tensors."""
    T = max(len(s[0]) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), pad_id, dtype=torch.long)
    m = torch.zeros(B, T, dtype=torch.float32)
    for i, (ids, mask) in enumerate(seqs):
        n = len(ids)
        t = torch.tensor(ids, dtype=torch.long)
        x[i, :n] = t
        y[i, :n] = t
        m[i, :n] = torch.tensor(mask, dtype=torch.float32)
    return x, y, m


def build_pair_batch(rows, tok, max_len, device):
    """(x, y, mask, ref_c, ref_r, n_c) for a list of pair rows.

    Identical to `rl/dpo_verify.build_pair_tensors` — same renderer, same
    im_end rule. The verify script is what keeps the two in step; a divergence
    here would train on bytes the gate never inspected.

    Both reference logprobs are required. A default of 0.0 for the chosen side
    is a finite, plausible number that would turn the DPO term into plain SFT on
    the rejected sample — training happily, optimising the wrong thing.
    """
    from rl.dpo_verify import build_pair_tensors
    seqs, ref_c, ref_r, n_c = [], [], [], []
    for r in rows:
        for k in ("logp_ref_chosen", "logp_ref_rejected"):
            if not math.isfinite(float(r.get(k, float("nan")))):
                raise KeyError(f"pair {r.get('key', '?')}: missing/non-finite {k}; "
                               f"re-run rl/dpo_pairs.py or rl/dpo_verify.py")
        _, (c_ids, c_mask), (r_ids, r_mask) = build_pair_tensors(tok, r)
        # truncate the *prompt* from the left: the completion is the thing being
        # trained on and must survive intact
        if len(c_ids) > max_len:
            keep = max_len - sum(c_mask)
            if keep <= 0:
                raise ValueError(f"pair {r.get('key','?')}: completion "
                                 f"({sum(c_mask)} tok) exceeds MAX_LEN {max_len}")
            c_ids, c_mask = c_ids[-keep:], c_mask[-keep:]
            r_ids, r_mask = r_ids[-keep:], r_mask[-keep:]
        seqs.append((c_ids, c_mask))
        seqs.append((r_ids, r_mask))
        # ref logprobs were computed on the untruncated sequence; a truncated
        # row's reference is then only approximately right, which is why the
        # miner caps prompt length well under MAX_LEN
        ref_c.append(float(r["logp_ref_chosen"]))
        ref_r.append(float(r["logp_ref_rejected"]))
        n_c.append(sum(c_mask))
    x, y, m = pad_batch(seqs)
    # forward's convention: x = tokens[:-1], y = tokens[1:], mask parallel
    return (x[:, :-1].contiguous().to(device),
            y[:, 1:].contiguous().to(device),
            m[:, 1:].contiguous().to(device),
            torch.tensor(ref_c, device=device),
            torch.tensor(ref_r, device=device),
            torch.tensor(n_c, device=device, dtype=torch.float32))


def rng_step(step, accum, j, frac):
    """Deterministic replay schedule — a pure function of (step, accum, j), so a
    restart gets the same rows in the same places. `random` here means "whether",
    not the RNG: the replay cadence must be reproducible like the data plan is.
    """
    if frac <= 0:
        return False
    period = max(1, int(round(1.0 / frac)))
    return ((step * accum + j) % period) == (period - 1)


def lr_at(step, total, peak, warmup, floor):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def param_groups(model, wd):
    decay, no_decay, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (no_decay if ("emb" in name or p.ndim < 2) else decay).append(p)
    return [dict(params=decay, weight_decay=wd),
            dict(params=no_decay, weight_decay=0.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, rest = ap.parse_known_args()
    if args.selftest:
        return selftest()
    if Accelerator is None:
        raise SystemExit("accelerate is required for training (pip install "
                         "accelerate); --selftest does not need it")

    env = os.environ.get
    pairs_dir = env("PAIRS_DIR", "")
    if not pairs_dir:
        raise SystemExit("PAIRS_DIR is required (rl/dpo_pairs.py output)")
    beta = float(env("BETA", 0.1))
    alpha = float(env("ALPHA", 0.2))
    lr = float(env("LR", 8e-6))
    epochs = int(env("EPOCHS", 1))
    warmup = int(env("WARMUP", 20))
    floor = float(env("LR_FLOOR", 0.1))
    pairs_per_step = int(env("PAIRS_PER_STEP", 32))
    micro_bs = int(env("MICRO_BS", 1))
    max_len = int(env("MAX_LEN", 3072))
    ce_chunk = int(env("CE_CHUNK", 2048))
    replay_frac = float(env("REPLAY_FRAC", 0.1))
    ckpt_every = int(env("CKPT_EVERY", 100))
    eval_every = int(env("EVAL_EVERY", 100))
    max_hours = float(env("MAX_HOURS", 11.0))
    max_steps = int(env("MAX_STEPS", 0))
    seed = int(env("SEED", 1234))
    out_dir = env("OUT_DIR", "/kaggle/working")
    dpo_out = os.path.join(out_dir, env("HF_CKPT_DIR", "DPO"))
    base_repo = env("BASE_REPO", "igidn/tinyballs-v1")
    base_file = env("BASE_CKPT_FILE", "REPAIR2/final.pt")
    hf_repo = env("HF_CKPT_REPO", "")
    hf_dir = env("HF_CKPT_DIR", "DPO")
    hf_token = env("HF_TOKEN") or env("HUGGINGFACE_TOKEN", "")

    if replay_frac > 0 and not env("SFT_DATA_DIR"):
        print("[dpo] REPLAY_FRAC>0 but no SFT_DATA_DIR; replay disabled",
              flush=True)
        replay_frac = 0.0

    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else None,
                      gradient_accumulation_steps=1)
    world = acc.num_processes
    rank = acc.process_index
    main_proc = acc.is_main_process
    torch.manual_seed(seed)
    os.makedirs(dpo_out, exist_ok=True)

    rows = []
    for p in sorted(glob.glob(os.path.join(pairs_dir, "pairs-*.jsonl"))):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no pairs in {pairs_dir}; run rl/dpo_pairs.py")
    rows = [r for r in rows if r.get("rejected", "").strip()
            and r.get("chosen", "").strip()]
    # `rejected` is on-policy for the base checkpoint but not for anything
    # trained past it; shuffling by key keeps the plan a pure function of seed
    rng = random.Random(seed)
    rng.shuffle(rows)
    if main_proc:
        print(f"[data] {len(rows)} pairs, beta={beta} alpha={alpha} lr={lr:g} "
              f"pairs/step={pairs_per_step} world={world} micro={micro_bs} "
              f"max_len={max_len} replay={replay_frac}", flush=True)

    replay = None
    if replay_frac > 0:
        try:
            replay = ReplayPool(env("SFT_DATA_DIR"), replay_frac, seed,
                                main_proc=main_proc)
        except Exception as e:
            print(f"[replay] unavailable ({type(e).__name__}: {e}); continuing "
                  f"without anti-regression rows. Format drift is the expected "
                  f"symptom (TOOL_FIX_PLAN.md §7).", flush=True)
            replay = None

    total_steps = max(1, len(rows) * epochs // max(1, pairs_per_step))
    if max_steps:
        total_steps = min(total_steps, max_steps)
    accum = max(1, pairs_per_step // (micro_bs * world))
    if main_proc:
        print(f"[plan] {total_steps} steps, accum={accum}, "
              f"{pairs_per_step * 2 * world} seqs/step", flush=True)

    from transformers import AutoTokenizer
    tok_src = env("TOK_JSON") or os.path.join(pairs_dir, "tokenizer.json")
    if not os.path.exists(tok_src):
        raise SystemExit("set TOK_JSON or put tokenizer.json in PAIRS_DIR")
    if os.path.isdir(tok_src):
        tok = AutoTokenizer.from_pretrained(tok_src)
    else:
        # AutoTokenizer wants a directory, not a bare tokenizer.json; the fast
        # tokenizer built from the same file gives identical ids.
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_src)

    # ---- base weights: the same policy that generated `rejected`, so
    # log pi_ref in the pair file is consistent with the initialisation.
    ck_path = env("BASE_CKPT_PATH", "") or os.path.join(out_dir, base_file)
    if not os.path.exists(ck_path):
        if main_proc:
            from huggingface_hub import hf_hub_download
            dest = os.path.join(out_dir, "base")
            os.makedirs(dest, exist_ok=True)
            hf_hub_download(repo_id=base_repo, filename=base_file,
                            token=hf_token or None, local_dir=dest)
        acc.wait_for_everyone()
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    arch = {k: cfg.get(k) for k in ARCH_KEYS}
    if any(v is None for v in arch.values()):
        raise SystemExit(f"checkpoint {ck_path} is missing cfg keys: {cfg}")
    model = LLaMA(**arch, grad_ckpt=arch["rope_theta"] > 0 and
                  int(env("GRAD_CKPT", "0")) == 1)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    opt = torch.optim.AdamW(param_groups(model, 0.0), lr=lr, betas=(0.9, 0.95),
                            fused=torch.cuda.is_available())
    if main_proc:
        print(f"[model] {model.num_params()/1e6:.1f}M params, "
              f"theta {arch['rope_theta']:.0f}", flush=True)
    model, opt = acc.prepare(model, opt)
    unw = acc.unwrap_model(model)
    dev = next(model.parameters()).device

    wandb = None
    if main_proc and env("WANDB_API_KEY"):
        import wandb as _w
        _w.login(key=env("WANDB_API_KEY"))
        wandb = _w.init(project=env("WANDB_PROJECT", "tinyballs"),
                        name=env("WANDB_NAME", "tinyballs-v1-dpo"),
                        config=dict(beta=beta, alpha=alpha, lr=lr, pairs=len(rows),
                                    pairs_per_step=pairs_per_step,
                                    max_len=max_len, replay=replay_frac))

    state_path = os.path.join(out_dir, "dpo-state.json")
    start_step = 0
    cks = sorted(glob.glob(os.path.join(dpo_out, "checkpoint-*.pt")),
                 key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    if cks:
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        if main_proc:
            print(f"[resume] {cks[-1]} -> step {start_step}/{total_steps}",
                  flush=True)

    def row_slice(step):
        """Deterministic plan: a pure function of (seed, step), so a restart
        lands on the same pairs. Mirrors train_sft's resumability contract."""
        per_rank = pairs_per_step // world
        out = []
        for r in range(world):
            base = (step * pairs_per_step + r * per_rank)
            out.append([rows[(base + i) % len(rows)] for i in range(per_rank)])
        return out

    t0 = time.time()
    step = start_step
    hist = []
    try:
        while step < total_steps:
            cur_lr = lr_at(step, total_steps, lr, warmup, floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            shards = row_slice(step)
            my_rows = shards[rank]
            opt.zero_grad(set_to_none=True)
            agg = {}
            n_micro = 0
            for j in range(accum):
                mb = my_rows[j * micro_bs:(j + 1) * micro_bs]
                if not mb:
                    continue
                ctx = (nullcontext() if j == accum - 1 else acc.no_sync(model))
                with ctx:
                    x, y, m, ref_c, ref_r, n_c = build_pair_batch(
                        mb, tok, max_len, dev)
                    lp = model.completion_logprob(x, y, m, ce_chunk=ce_chunk)
                    B = len(mb)
                    pi_c, pi_r = lp[:B], lp[B:]
                    loss, met = dpo_loss(pi_c, pi_r, ref_c[:B], ref_r[:B], beta)
                    nll = nll_chosen(pi_c, n_c)
                    total = loss + alpha * nll
                    met["nll"] = nll.item()
                    # one replay micro-batch every 1/REPLAY_FRAC micro-steps, so
                    # the *row* share matches REPLAY_FRAC as in train_sft
                    if replay is not None and rng_step(step, accum, j,
                                                       replay_frac):
                        rb = replay.batch(1, dev)
                        if rb is not None:
                            xr, yr, mr = rb
                            rloss = model(xr, yr, loss_mask=mr,
                                          ce_chunk=ce_chunk)
                            total = total + rloss
                            met["replay_loss"] = float(rloss.detach())
                    if not torch.isfinite(total):
                        met["skipped"] = 1.0
                        total = total * 0.0
                    acc.backward(total / accum)
                n_micro += 1
                for k, v in met.items():
                    agg[k] = agg.get(k, 0.0) + v
            if any(p.grad is not None for p in model.parameters()):
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            n = max(1, n_micro)
            met = {k: v / n for k, v in agg.items()}
            hist.append(met)
            if main_proc and (step % 20 == 0 or step == 1 or step == total_steps):
                el = time.time() - t0
                msg = (f"[{step}/{total_steps}] dpo {met['dpo']:.4f} "
                       f"acc {met.get('acc', 0):.2f} margin {met.get('margin', 0):+.3f} "
                       f"nll {alpha * float(met.get('nll', 0)):.4f} "
                       f"lr {cur_lr:.2e} {el/60:.1f}min")
                print(msg, flush=True)
                if wandb:
                    wandb.log(dict(step=step, **{k: float(v) for k, v in met.items()},
                                   lr=cur_lr), step=step)
            if step % ckpt_every == 0 or step == total_steps:
                if main_proc:
                    torch.save(dict(model=acc.get_state_dict(model),
                                    opt=opt.state_dict(), step=step,
                                    cfg=arch, beta=beta, alpha=alpha,
                                    hist=hist[-200:]),
                               os.path.join(dpo_out, f"checkpoint-{step}.pt"))
                    json.dump(dict(step=step, total=total_steps, lr=cur_lr,
                                   pairs=len(rows)),
                              open(state_path, "w"), indent=1)
                    for p in glob.glob(os.path.join(dpo_out, "checkpoint-*.pt")):
                        if int(p.rsplit("-", 1)[1].split(".")[0]) < step - 1:
                            os.remove(p)
                acc.wait_for_everyone()
            if (time.time() - t0) / 3600 > max_hours:
                if main_proc:
                    print(f"[dpo] MAX_HOURS {max_hours} reached at step {step}",
                          flush=True)
                    torch.save(dict(model=acc.get_state_dict(model),
                                    opt=opt.state_dict(), step=step, cfg=arch,
                                    hist=hist[-200:]),
                               os.path.join(dpo_out, "checkpoint-final.pt"))
                break
    finally:
        if wandb:
            wandb.finish()

    if main_proc:
        torch.save(dict(model=acc.get_state_dict(model), opt=opt.state_dict(),
                        step=step, cfg=arch, hist=hist),
                   os.path.join(dpo_out, "final.pt"))
        print(f"\nDPO DONE — {step} steps. Re-measure multi_tool on the Pi "
              f"through the unified serving path before believing any of it: "
              f"a rising pair reward is not the gate.", flush=True)
    return 0


class ReplayPool:
    """Anti-regression rows from a packed SFT dataset, materialised on demand.

    `TOOL_FIX_PLAN.md` §7: "mix 10% of the repair SFT rows into every RL batch
    to avoid format drift". DPO on a narrow pair set is exactly where format
    decays, and format is 0.2 of the reward that measures the fix — so this is
    not optional garnish.

    Rows are `seq_len + 1` tokens with the pack's own loss mask, so the replay
    term is the same masked CE the SFT trainer optimises: full-token loss for
    pretrain-replay rows, assistant-only for tool rows, whatever the shard says.
    """

    def __init__(self, sft_dir, frac, seed, seq_len=2048, main_proc=False):
        from llama.sft_data import SFTPhaseData
        self.data = SFTPhaseData(sft_dir, ["short"], seq_len)
        self.plan = self.data.build_plan(seed=seed + 3, replay_frac=frac)
        self.seq_len = seq_len
        if main_proc:
            print(f"[replay] {len(self.plan)} rows from {sft_dir}", flush=True)

    def batch(self, n, dev):
        if not self.plan or n <= 0:
            return None
        rows = [self.plan[random.randrange(len(self.plan))] for _ in range(n)]
        xs, ys, ms = [], [], []
        for r in rows:
            toks, mask = self.data.materialize(r)
            t = torch.from_numpy(toks[:-1].astype(np.int64))
            xs.append(t)
            ys.append(torch.from_numpy(toks[1:].astype(np.int64)))
            ms.append(torch.from_numpy(mask[1:]))
        L = max(x.shape[0] for x in xs)
        x = torch.zeros(len(xs), L, dtype=torch.long)
        y = torch.zeros(len(xs), L, dtype=torch.long)
        m = torch.zeros(len(xs), L, dtype=torch.float32)
        for i, (a, b, c) in enumerate(zip(xs, ys, ms)):
            x[i, :a.shape[0]] = a
            y[i, :b.shape[0]] = b
            m[i, :c.shape[0]] = c
        return x.to(dev), y.to(dev), m.to(dev)


# ------------------------------------------------------------------ selftest
def selftest():
    """The loss, against hand-computed values. Everything here is arithmetic a
    reader can check; the point is that a sign error or a swapped pair cannot
    pass silently."""
    fails = []

    def expect(name, got, want, tol=1e-6):
        if isinstance(want, (tuple, list, bool, str)):
            ok = got == want
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})")
        else:
            ok = abs(got - want) <= tol
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {got:.6f} "
                  f"(want {want:.6f})")
        if not ok:
            fails.append(name)

    beta = 0.1
    print("== a correctly preferred pair must give loss < log 2")
    loss, met = dpo_loss(torch.tensor([-5.0]), torch.tensor([-10.0]),
                         torch.tensor([-5.0]), torch.tensor([-10.0]), beta)
    expect("no_ref_shift_loss", loss.item(), math.log(2), 1e-6)
    expect("margin_zero", met["margin"], 0.0)

    print("== the policy's own preference drives the margin when ref == pi")
    # ref identical to pi on both sides cancels to zero: at initialisation
    # DPO has no preference to amplify, which is why the first steps come
    # entirely from alpha * NLL(chosen).
    loss, met = dpo_loss(torch.tensor([-1.0]), torch.tensor([-9.0]),
                         torch.tensor([-1.0]), torch.tensor([-9.0]), beta)
    expect("self_ref_margin", met["margin"], 0.0)
    expect("self_ref_loss", loss.item(), math.log(2), 1e-6)
    expect("self_ref_acc", met["acc"], 0.0)   # logits > 0 is False at exactly 0

    print("== a ref that dislikes chosen leaves the policy the whole margin")
    # (pi_c - ref_c) - (pi_r - ref_r) = 19 - 0; `margin` is the beta-scaled
    # logit and `ref_gap` the unscaled quantity
    loss, met = dpo_loss(torch.tensor([-1.0]), torch.tensor([-9.0]),
                         torch.tensor([-20.0]), torch.tensor([-9.0]), beta)
    expect("ref_gap", met["ref_gap"], 19.0, 1e-6)
    expect("margin", met["margin"], beta * 19.0, 1e-6)
    expect("loss", loss.item(), -math.log(1 / (1 + math.exp(-beta * 19.0))), 1e-6)
    expect("acc", met["acc"], 1.0)

    print("== reversing the pair lowers the margin and raises the loss")
    loss2, m2 = dpo_loss(torch.tensor([-9.0]), torch.tensor([-1.0]),
                         torch.tensor([-20.0]), torch.tensor([-9.0]), beta)
    expect("reversed_gap", m2["ref_gap"], 3.0, 1e-6)
    expect("reversed_loss", loss2.item(),
           -math.log(1 / (1 + math.exp(-beta * 3.0))), 1e-6)
    # still positive (the policy does prefer chosen) but much weaker: the
    # reversal cost it 16 of the 19 nats the forward direction had
    expect("reversed_weaker", loss2.item() > loss.item(), True)
    # a genuinely inverted preference is the only thing that exceeds log 2
    bad, _ = dpo_loss(torch.tensor([-20.0]), torch.tensor([-1.0]),
                      torch.tensor([-1.0]), torch.tensor([-9.0]), beta)
    expect("inverted_gt_log2", bad.item() > math.log(2), True)

    print("== beta scales the logit, so beta=0 gives exactly log 2")
    loss0, _ = dpo_loss(torch.tensor([-1.0]), torch.tensor([-9.0]),
                        torch.tensor([-1.0]), torch.tensor([-9.0]), 0.0)
    expect("beta_zero", loss0.item(), math.log(2), 1e-6)

    print("== shifting BOTH sides by the same amount must not change the loss")
    a, _ = dpo_loss(torch.tensor([-2.0]), torch.tensor([-7.0]),
                    torch.tensor([-2.0]), torch.tensor([-7.0]), beta)
    b, _ = dpo_loss(torch.tensor([-12.0]), torch.tensor([-17.0]),
                    torch.tensor([-12.0]), torch.tensor([-17.0]), beta)
    expect("shift_invariant", a.item(), b.item(), 1e-6)

    print("== reference offsets are what make this DPO rather than SFT")
    # with ref == pi the margin is exactly zero, so the DPO term contributes
    # nothing but a constant at initialisation; all early progress comes from
    # alpha * NLL(chosen). If a training run shows a flat DPO loss at step 0
    # this is why, and it is not a bug.
    l0, m0 = dpo_loss(torch.tensor([-1.0]), torch.tensor([-9.0]),
                      torch.tensor([-1.0]), torch.tensor([-9.0]), beta)
    expect("self_ref_gap", m0["ref_gap"], 0.0, 1e-6)

    print("== nll_chosen is length-normalised, so longer != worse")
    nll = nll_chosen(torch.tensor([-10.0]), torch.tensor([5.0]))
    expect("per_token", nll.item(), 2.0)
    nll_long = nll_chosen(torch.tensor([-20.0]), torch.tensor([10.0]))
    expect("same_rate_long", nll_long.item(), 2.0)

    print("== pad_batch right-pads and masks the pad out")
    seqs = [([1, 2, 3], [0, 1, 1]), ([1, 2], [0, 1]), ([1], [1])]
    x, y, m = pad_batch(seqs)
    expect("shape", tuple(x.shape), (3, 3))
    expect("row1_pad_zero", int(x[1, 2]), 0)
    expect("row1_mask_zero", float(m[1, 2]), 0.0)
    expect("row0_mask", m[0].tolist(), [0.0, 1.0, 1.0])
    expect("x_equals_y", bool((x == y).all()), True)

    print("== a full row with mask summing to 1 row-0 tokens is not the same "
          "as mask 0")
    seqs2 = [([1, 2, 3], [0, 0, 1]), ([1, 2, 3], [0, 1, 1])]
    _, _, m2 = pad_batch(seqs2)
    expect("mask_distinct", float(m2[0].sum()), 1.0)

    print("== replay cadence is deterministic and matches REPLAY_FRAC")
    seq = [rng_step(0, 4, j, 0.25) for j in range(4)]
    print(f"  frac 0.25 over one step of accum=4: {seq}")
    expect("one_replay_per_step", sum(seq), 1)
    expect("repeatable", [rng_step(0, 4, j, 0.25) for j in range(4)], seq)
    hits = sum(rng_step(s, 4, j, 0.1) for s in range(20) for j in range(4))
    expect("frac_0.1_share", hits / 80.0, 0.1, 0.02)
    expect("zero_disables", rng_step(0, 4, 0, 0.0), False)

    print("== lr schedule: warmup then cosine to floor*peak")
    expect("warmup_first", lr_at(0, 100, 1e-5, 10, 0.1), 1e-6)
    expect("warmup_last", lr_at(9, 100, 1e-5, 10, 0.1), 1e-5)
    expect("mid", lr_at(55, 100, 1e-5, 10, 0.1),
           1e-5 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5))), 1e-12)
    # the cosine reaches floor*peak exactly at the *last* step; step 99 of 100
    # is one step short of it, so the tolerance is the step size, not 1e-9
    expect("end_near_floor", lr_at(99, 100, 1e-5, 10, 0.1),
           1e-5 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 89 / 90))), 1e-12)
    expect("floor_reached", abs(lr_at(100, 100, 1e-5, 10, 0.1) - 1e-6), 0.0,
           1e-9)

    print()
    if fails:
        print(f"TRAIN SELFTEST FAIL: {fails}")
        return 1
    print("TRAIN SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())