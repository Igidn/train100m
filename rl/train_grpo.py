#!/usr/bin/env python
"""GRPO trainer for TinyBalls-V1 — RL plan 2 stage B (`rl/RL_PLAN_2.md` §3, §8.1).

One on-policy gradient step per sampled batch. The samples are exactly on-policy
(G rollouts of the current policy, `rl/sample_pool.py`), so there is no ratio and
no clipping: the PPO ratio is 1 by construction. The loss is DAPO-shaped:

    pg  = -(A_i .detach() * logpi_i).sum() / total_tokens        # token-level
    r   = (logpi_ref - logpi) / n_i                              # per-token mean
    kl  = (K3(r) * n_i).sum() / total_tokens,  K3(r)=exp(r)-r-1
    loss = pg + BETA_KL * kl

Three things this file is careful about, each a §3.1 trap:

  1. **Whole groups per rank.** The sample file stores group-normalised
     advantages, and `group_chunk`/`rank_rows` hand each group to exactly one
     rank, so no rank ever averages half a group.
  2. **Token-level, not per-sequence.** The denominator is the step's total
     completion tokens, shared across micro-batches. A per-sequence mean would
     bias toward short completions — the same trap `nll_chosen` documents in
     `train_dpo.py`.
  3. **The advantage is detached data.** `A_i` never carries a gradient; the loss
     graph is `logpi` alone, hanging off `DistributedDataParallel.forward`
     (`mode="logprob"`), which is the only side DDP allreduces.

`logpi_ref` is a constant column (`rl/refpass.py`) for the whole round: the
reference is the round's starting policy, recomputed each round. Like Stage A, no
reference model is carried in memory. Best-of-N -> DPO stays the fallback
(RL_PLAN_2 §8.2) if this misbehaves; `rl/train_dpo.py` is unchanged either way.

Env (all overridable):
  SAMPLES_DIR    rl/sample_pool.py output dir (required)
  BETA_KL        0.02     GRPO LR is the knob that blew up before — start low
  LR             5e-6      and read a fast KL rise with a flat proxy as the failure
  EPOCHS 1  ROWS_PER_STEP 32  MICRO_BS 1  MAX_LEN 3072  CE_CHUNK 2048
  REPLAY_FRAC 0.1  MAX_HOURS 11  CKPT_EVERY 100  OUT_DIR  SEED
  BASE_REPO BASE_CKPT_FILE BASE_CKPT_PATH

    # on the torch box, from the round's starting policy:
    python rl/refpass.py --samples samples/batch-0001.jsonl \\
        --out samples/batch-0001.jsonl --base-ckpt-path /path/final.pt
    SAMPLES_DIR=samples python rl/train_grpo.py
    python rl/train_grpo.py --selftest        # needs torch (use the 3.12 venv)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import nullcontext

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llama.model import LLaMA  # noqa: E402
from rl.dpo_verify import IM_END_ID  # noqa: E402
from sft.rollout import render_prefix  # noqa: E402
# shared with the DPO trainer so the two cannot drift: padding, LR schedule,
# optimiser groups, the deterministic replay cadence, and the replay pool.
from rl.train_dpo import ReplayPool, lr_at, pad_batch, param_groups, rng_step  # noqa: E402

try:
    from accelerate import Accelerator
except ImportError:      # pragma: no cover - selftest path
    Accelerator = None

ARCH_KEYS = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
             "head_dim", "ffn_dim", "rope_theta", "rms_eps")

KL_CLAMP = 20.0


# ------------------------------------------------------------------ loss
def kl_k3(logpi, logpi_ref, n, clamp=KL_CLAMP):
    """Per-row K3 KL term, `exp(r) - r - 1`, r = mean per-token log(q/p).

    `logpi`/`logpi_ref` are the *summed* logprobs the model returns
    (`completion_logprob`), so the ratio is taken per token by dividing by `n`.
    K3 is non-negative and low-variance; the clamp only guards `exp` on a wild
    early step, and a clamp at +/-20 is far outside any honest ratio.
    """
    r = ((logpi_ref - logpi) / n.clamp(min=1.0)).clamp(-clamp, clamp)
    return torch.exp(r) - r - 1.0


def grpo_loss(logpi, logpi_ref, adv, n, beta_kl, total_tokens, clamp=KL_CLAMP):
    """(loss, metrics). `total_tokens` is the *step's* completion-token count."""
    denom = max(1.0, float(total_tokens))
    adv = adv.detach()
    pg = -(adv * logpi).sum() / denom
    kl_terms = kl_k3(logpi, logpi_ref, n, clamp)
    kl = (kl_terms * n).sum() / denom
    loss = pg + beta_kl * kl
    return loss, {
        "loss": loss.item(),
        "pg": pg.item(),
        "kl": kl.item(),
        "adv_abs_mean": adv.abs().mean().item(),
        "adv_abs_max": adv.abs().max().item(),
        # the entropy proxy we can afford: mean per-token logprob. Higher = more
        # confident (less entropy); a jump here is the collapse signature (§12).
        "mean_logp": (logpi.detach() / n.clamp(min=1)).mean().item(),
        "n_rows": int(logpi.shape[0]),
    }


def build_grpo_batch(rows, tok, max_len, device):
    """(x, y, mask, adv, logpi_ref, n) for a list of turn rows.

    Identical sequence construction to `rl/dpo_verify.build_pair_tensors` for one
    side, and to `rl/refpass.sample_sides`: `render_prefix(prompt_msgs) +
    encode(completion) + [im_end]` iff the turn stopped cleanly. If this drifts
    from the ref pass, the KL measures the wrong origin.
    """
    seqs, adv, ref, n = [], [], [], []
    for r in rows:
        v = r.get("logpi_ref")
        if v is None or not math.isfinite(float(v)):
            raise KeyError(f"row {r.get('key', '?')}: missing logpi_ref; run rl/refpass.py")
        prompt_ids = render_prefix(tok, r["prompt_msgs"])
        body = tok.encode(r["completion"], add_special_tokens=False)
        body = list(getattr(body, "ids", body))
        if r.get("completion_stopped", True):
            body = body + [IM_END_ID]
        if not body:
            raise ValueError(f"row {r.get('key','?')}: empty completion")
        ids = prompt_ids + body
        mask = [0] * len(prompt_ids) + [1] * len(body)
        if len(ids) > max_len:
            keep = max_len - sum(mask)      # truncate the prompt, never the turn
            if keep <= 0:
                raise ValueError(f"row {r.get('key','?')}: completion "
                                 f"({sum(mask)} tok) exceeds MAX_LEN {max_len}")
            ids, mask = ids[-keep:], mask[-keep:]
        seqs.append((ids, mask))
        adv.append(float(r["advantage"]))
        ref.append(float(r["logpi_ref"]))
        n.append(sum(mask))
    x, y, m = pad_batch(seqs)
    return (x[:, :-1].contiguous().to(device),
            y[:, 1:].contiguous().to(device),
            m[:, 1:].contiguous().to(device),
            torch.tensor(adv, device=device),
            torch.tensor(ref, device=device),
            torch.tensor(n, device=device, dtype=torch.float32))


# ------------------------------------------------------------------ group plan
def group_chunk(groups, step, groups_needed):
    """The `groups_needed` groups for a step — a pure function of (step, plan),
    so a restart lands on the same groups (train_dpo's resumability contract)."""
    n = len(groups)
    base = (step * groups_needed) % n
    return [groups[(base + i) % n] for i in range(groups_needed)]


def rank_rows(by_group, chunk, rank, world):
    """The rows a rank owns this step. Whole groups only (§3.1 trap 1)."""
    mine = [chunk[i] for i in range(len(chunk)) if i % world == rank]
    return [r for g in mine for r in by_group[g]]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args, rest = ap.parse_known_args()
    if args.selftest:
        return selftest()
    if Accelerator is None:
        raise SystemExit("accelerate is required for training (pip install accelerate); "
                         "--selftest does not need it")

    env = os.environ.get
    samples_dir = env("SAMPLES_DIR", "")
    if not samples_dir:
        raise SystemExit("SAMPLES_DIR is required (rl/sample_pool.py output)")
    beta_kl = float(env("BETA_KL", 0.02))
    lr = float(env("LR", 5e-6))
    epochs = int(env("EPOCHS", 1))
    warmup = int(env("WARMUP", 20))
    floor = float(env("LR_FLOOR", 0.1))
    rows_per_step = int(env("ROWS_PER_STEP", env("PAIRS_PER_STEP", 32)))
    micro_bs = int(env("MICRO_BS", 1))
    max_len = int(env("MAX_LEN", 3072))
    ce_chunk = int(env("CE_CHUNK", 2048))
    replay_frac = float(env("REPLAY_FRAC", 0.1))
    ckpt_every = int(env("CKPT_EVERY", 100))
    max_hours = float(env("MAX_HOURS", 11.0))
    max_steps = int(env("MAX_STEPS", 0))
    seed = int(env("SEED", 1234))
    out_dir = env("OUT_DIR", "/kaggle/working")
    grpo_out = os.path.join(out_dir, env("HF_CKPT_DIR", "GRPO"))
    base_repo = env("BASE_REPO", "igidn/tinyballs-v1")
    base_file = env("BASE_CKPT_FILE", "REPAIR2/final.pt")
    hf_token = env("HF_TOKEN") or env("HUGGINGFACE_TOKEN", "")

    if replay_frac > 0 and not env("SFT_DATA_DIR"):
        print("[grpo] REPLAY_FRAC>0 but no SFT_DATA_DIR; replay disabled", flush=True)
        replay_frac = 0.0

    acc = Accelerator(mixed_precision="fp16" if torch.cuda.is_available() else None,
                      gradient_accumulation_steps=1)
    world, rank, main_proc = acc.num_processes, acc.process_index, acc.is_main_process
    torch.manual_seed(seed)
    os.makedirs(grpo_out, exist_ok=True)

    # ---- data: turn rows, grouped by episode, degenerate groups dropped
    rows = []
    for p in sorted(glob.glob(os.path.join(samples_dir, "**", "*.jsonl"), recursive=True)):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {samples_dir}; run rl/sample_pool.py")
    rows = [r for r in rows if (r.get("completion") or "").strip()
            and not r.get("degenerate")]
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r)
    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    if not groups:
        raise SystemExit("every group is degenerate; the ladder is dead (see G0)")
    avg_rows = len(rows) / len(groups)
    groups_per_rank = max(1, int(math.ceil((rows_per_step // max(1, world)) / avg_rows)))
    groups_needed = groups_per_rank * world
    steps_per_epoch = max(1, len(groups) // groups_needed)
    total_steps = max(1, steps_per_epoch * epochs)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    if main_proc:
        print(f"[data] {len(rows)} rows / {len(groups)} live groups "
              f"(avg {avg_rows:.1f} rows/group, degenerate dropped) "
              f"beta_kl={beta_kl} lr={lr:g} rows/step={rows_per_step} world={world} "
              f"micro={micro_bs} max_len={max_len} replay={replay_frac}", flush=True)
        print(f"[plan] {total_steps} steps, {groups_needed} groups/step "
              f"({groups_per_rank}/rank), {world} rank(s)", flush=True)

    replay = None
    if replay_frac > 0:
        try:
            replay = ReplayPool(env("SFT_DATA_DIR"), replay_frac, seed,
                                main_proc=main_proc)
        except Exception as e:
            print(f"[replay] unavailable ({type(e).__name__}: {e}); continuing "
                  f"without anti-regression rows (TOOL_FIX_PLAN.md §7).", flush=True)
            replay = None

    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    tok_src = env("TOK_JSON") or os.path.join(samples_dir, "tokenizer.json")
    if not os.path.exists(tok_src):
        raise SystemExit("set TOK_JSON or put tokenizer.json in SAMPLES_DIR")
    tok = (AutoTokenizer.from_pretrained(tok_src) if os.path.isdir(tok_src)
           else PreTrainedTokenizerFast(tokenizer_file=tok_src))

    # ---- base weights: the round's starting policy (Stage-A output for round 1)
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
    model = LLaMA(**arch, grad_ckpt=int(env("GRAD_CKPT", "0")) == 1)
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
                        name=env("WANDB_NAME", "tinyballs-v1-grpo"),
                        config=dict(beta_kl=beta_kl, lr=lr, rows=len(rows),
                                    groups=len(groups), rows_per_step=rows_per_step,
                                    max_len=max_len, replay=replay_frac))

    state_path = os.path.join(out_dir, "grpo-state.json")
    start_step = 0
    cks = sorted(glob.glob(os.path.join(grpo_out, "checkpoint-*.pt")),
                 key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    if cks:
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        unw.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        if main_proc:
            print(f"[resume] {cks[-1]} -> step {start_step}/{total_steps}", flush=True)

    t0 = time.time()
    step = start_step
    hist = []
    try:
        while step < total_steps:
            cur_lr = lr_at(step, total_steps, lr, warmup, floor)
            for g in opt.param_groups:
                g["lr"] = cur_lr
            chunk = group_chunk(groups, step, groups_needed)
            my_rows = rank_rows(by_group, chunk, rank, world)
            if not my_rows:
                step += 1
                continue

            accum = max(1, int(math.ceil(len(my_rows) / micro_bs)))
            mbs, step_tokens = [], 0.0
            for j in range(accum):
                mb = my_rows[j * micro_bs:(j + 1) * micro_bs]
                if not mb:
                    continue
                try:
                    tensors = build_grpo_batch(mb, tok, max_len, dev)
                except (KeyError, ValueError) as e:
                    print(f"[warn] skipping rows: {e}", flush=True)
                    continue
                mbs.append(tensors)
                step_tokens += float(tensors[5].sum())
            if not mbs:
                step += 1
                continue

            opt.zero_grad(set_to_none=True)
            agg, n_micro = {}, 0
            for j, (x, y, m, adv, ref, n) in enumerate(mbs):
                ctx = (nullcontext() if j == len(mbs) - 1 else acc.no_sync(model))
                with ctx:
                    # via forward(..., mode="logprob"): the graph that DDP
                    # allreduces is the one hanging off its own output.
                    lp = model(x, y, loss_mask=m, ce_chunk=ce_chunk, mode="logprob")
                    loss, met = grpo_loss(lp, ref, adv, n, beta_kl, step_tokens)
                    if replay is not None and rng_step(step, accum, j, replay_frac):
                        rb = replay.batch(1, dev)
                        if rb is not None:
                            xr, yr, mr = rb
                            rloss = model(xr, yr, loss_mask=mr, ce_chunk=ce_chunk)
                            loss = loss + rloss
                            met["replay_loss"] = float(rloss.detach())
                    if not torch.isfinite(loss):
                        met["skipped"] = 1.0
                        loss = loss * 0.0
                    acc.backward(loss / len(mbs))
                n_micro += 1
                for k, v in met.items():
                    agg[k] = agg.get(k, 0.0) + v
            if any(p.grad is not None for p in model.parameters()):
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            met = {k: v / max(1, n_micro) for k, v in agg.items()}
            met["step"] = step
            hist.append(met)
            if main_proc and (step % 10 == 0 or step == 1 or step == total_steps):
                el = time.time() - t0
                print(f"[{step}/{total_steps}] loss {met.get('loss',0):+.4f} "
                      f"pg {met.get('pg',0):+.4f} kl {met.get('kl',0):.4f} "
                      f"mean_logp {met.get('mean_logp',0):+.3f} "
                      f"|A| {met.get('adv_abs_mean',0):.3f} lr {cur_lr:.2e} "
                      f"{el/60:.1f}min", flush=True)
                if wandb:
                    wandb.log(dict({k: float(v) for k, v in met.items()}, lr=cur_lr),
                              step=step)
            if step % ckpt_every == 0 or step == total_steps:
                if main_proc:
                    torch.save(dict(model=acc.get_state_dict(model),
                                    opt=opt.state_dict(), step=step, cfg=arch,
                                    beta_kl=beta_kl, hist=hist[-400:]),
                               os.path.join(grpo_out, f"checkpoint-{step}.pt"))
                    json.dump(dict(step=step, total=total_steps, lr=cur_lr,
                                   groups=len(groups), rows=len(rows),
                                   hist=hist[-400:]),
                              open(state_path, "w"), indent=1)
                    for p in glob.glob(os.path.join(grpo_out, "checkpoint-*.pt")):
                        if int(p.rsplit("-", 1)[1].split(".")[0]) < step - 1:
                            os.remove(p)
                acc.wait_for_everyone()
            if (time.time() - t0) / 3600 > max_hours:
                if main_proc:
                    print(f"[grpo] MAX_HOURS {max_hours} reached at step {step}", flush=True)
                    torch.save(dict(model=acc.get_state_dict(model),
                                    opt=opt.state_dict(), step=step, cfg=arch,
                                    hist=hist[-400:]),
                               os.path.join(grpo_out, "checkpoint-final.pt"))
                break
    finally:
        if wandb:
            wandb.finish()

    if main_proc:
        torch.save(dict(model=acc.get_state_dict(model), opt=opt.state_dict(),
                        step=step, cfg=arch, hist=hist[-400:]),
                   os.path.join(grpo_out, "final.pt"))
        print(f"\nGRPO DONE — {step} steps. Re-measure behaviour on the held-out "
              f"split through the unified serving path (RL_PLAN_2 §9 G4) before "
              f"believing any of it: a falling pg is not the gate. Watch kl and "
              f"mean_logp for the collapse signature (§12).", flush=True)
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    """The loss and the group plan against hand-computable values."""
    fails = []

    def expect(name, got, want, tol=1e-5):
        if isinstance(want, bool) or not isinstance(want, (int, float)):
            ok = (got == want)
        else:
            ok = abs(got - want) <= tol
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})")
        if not ok:
            fails.append(name)

    n = torch.tensor([4.0, 4.0])
    print("== pg: zero advantage contributes nothing; sign is a descent step")
    lp = torch.tensor([-10.0, -10.0], requires_grad=True)
    loss, met = grpo_loss(lp, lp.detach(), torch.zeros(2), n, 0.0, 8.0)
    expect("pg_zero_adv", met["pg"], 0.0)
    loss.backward()
    expect("grad_zero_adv", lp.grad.abs().sum().item(), 0.0)
    lp2 = torch.tensor([-10.0], requires_grad=True)
    grpo_loss(lp2, torch.tensor([-10.0]), torch.tensor([1.0]), torch.tensor([1.0]),
              0.0, 1.0)[0].backward()
    expect("grad_negative_for_pos_adv", lp2.grad.item() < 0, True)

    print("== kl: zero when policy == ref, non-negative always")
    lp = torch.tensor([-8.0, -3.0])
    l0, m0 = grpo_loss(lp, lp.clone(), torch.tensor([0.0, 0.0]), n, 1.0, 8.0)
    expect("kl_zero_at_ref", m0["kl"], 0.0)
    # policy BELOW ref on row 0 (logpi < ref) -> r>0 -> k3>0
    l1, m1 = grpo_loss(torch.tensor([-20.0]), torch.tensor([-4.0]),
                       torch.tensor([0.5]), torch.tensor([4.0]), 1.0, 4.0)
    expect("kl_positive_when_below_ref", m1["kl"] > 0, True)
    # k3 >= 0 for every r, including r<0
    r = torch.linspace(-5, 5, 21)
    k3 = torch.exp(r) - r - 1.0
    expect("k3_nonneg", bool((k3 >= -1e-6).all()), True)

    print("== token-level: the same per-token logp at two lengths is unbiased")
    # row A: 2 tokens at -1 each; row B: 6 tokens at -1 each. Per-*sequence* mean
    # would weight them equally; token-level weights B three times.
    lp = torch.tensor([-2.0, -6.0])
    _, m = grpo_loss(lp, lp.clone(), torch.tensor([1.0, 1.0]),
                     torch.tensor([2.0, 6.0]), 0.0, 8.0)
    # pg = -(-2 -6)/8 = 1.0 ; if it were per-sequence it would be -(-1 -1)/2 = 1.0
    # too, so use unequal advantages to separate the two
    _, ma = grpo_loss(torch.tensor([-2.0, -6.0]), torch.tensor([-2.0, -6.0]),
                      torch.tensor([1.0, 0.0]), torch.tensor([2.0, 6.0]), 0.0, 8.0)
    expect("pg_token_level", ma["pg"], 2.0 / 8.0)

    print("== group plan: whole groups, disjoint across ranks, deterministic")
    by_group = {f"t{i}:g0": [{"k": f"t{i}:g0:{j}"} for j in range(2 + i % 3)]
                for i in range(8)}
    groups = sorted(by_group)
    chunk = group_chunk(groups, 0, 4)
    expect("chunk_size", len(chunk), 4)
    r0 = rank_rows(by_group, chunk, 0, 2)
    r1 = rank_rows(by_group, chunk, 1, 2)
    ok0 = {r["k"].rsplit(":", 1)[0] for r in r0}
    ok1 = {r["k"].rsplit(":", 1)[0] for r in r1}
    expect("disjoint", len(ok0 & ok1), 0)
    expect("union_is_chunk", ok0 | ok1, set(chunk))
    expect("deterministic", group_chunk(groups, 0, 4), chunk)
    expect("advances", group_chunk(groups, 1, 4) != chunk, True)

    print("== build_grpo_batch + a real forward on a toy LLaMA (mask, grads)")
    tok_path = next((c for c in [
        os.path.expanduser("~/.cache/tinyballs-import/tokenizer.json"),
        os.path.join(_ROOT, "toktest", "tokenizer.json")] if os.path.exists(c)), "")
    if not tok_path:
        print("  SKIP (no tokenizer)")
    else:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        from build_pairs import HEADER, TOOL_CALL
        row = {"key": "s", "prompt_msgs": [{"role": "system", "content": "S"},
                                           {"role": "user", "content": "U"}],
               "completion": HEADER + TOOL_CALL + '{"name": "bash", "arguments": {"command": "ls"}}',
               "completion_stopped": True, "advantage": 1.0, "logpi_ref": -7.5}
        x, y, m, adv, ref, nn = build_grpo_batch([row], tok, 512, "cpu")
        expect("batch_shapes", (x.shape[0], y.shape[0], m.shape[0]), (1, 1, 1))
        expect("mask_only_completion", float(m.sum()), nn.item())
        expect("prompt_unmasked", float(m[0, 0]), 0.0)
        arch = dict(vocab_size=49154, dim=32, n_layers=1, n_heads=4, n_kv_heads=4,
                    head_dim=8, ffn_dim=64, rope_theta=10000.0, rms_eps=1e-5)
        model = LLaMA(**arch)
        lp = model(x, y, loss_mask=m, ce_chunk=128, mode="logprob")
        expect("logprob_shape", tuple(lp.shape), (1,))
        loss, met = grpo_loss(lp, ref, adv, nn, 0.02, float(nn.sum()))
        loss.backward()
        g = [p.grad for p in model.parameters() if p.grad is not None]
        expect("grads_flow", len(g) > 0, True)

    print()
    if fails:
        print(f"GRPO SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("GRPO SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
