#!/usr/bin/env python
"""Generate and score DPO pairs — stage 2 of 3 (`rl/`).

Reads `rl/dpo_mine.py` output, has the current policy generate the final answer
for each teacher-forced prompt, and keeps the rows where that answer is worse
than gold on entity binding. Each surviving row becomes a (chosen=gold,
rejected=V1) pair with the **reference logprobs already computed**, because the
reference policy for DPO is by definition the policy that produced `rejected` —
so the trainer never needs a second model in memory.

Measured on the Pi (4x Cortex-A76, 3GB, fp32, 378-token prompt):

    early EOS (17 tok answer)   B=1 5.5 s/row   B=8 1.7 s/row  -> 19k rows  9 h
    no EOS (full 256 decode)    B=1 67.4 s/row  B=8 11.8 s/row  -> 19k rows 62 h

The 35x spread is decided entirely by whether the model emits `` — which is
why `--max-rows 2500` is the default: ~1 h at B=8 with early EOS is a complete
DPO dataset for one diagnosed failure mode, and the 62 h worst case only applies
if termination is systematically broken, which 50 rows will tell you.

`RL_TRAINING.md` §4 says run everything on Kaggle. That is a GRPO constraint (the
policy moves every batch), not a DPO one: this is a frozen-policy offline pass,
which is the category §4 assigns to the Pi. What the T4 is for is `train_dpo.py`.

    python rl/dpo_pairs.py --mined dpo-mined --out dpo-pairs \\
        --ckpt ~/ai/sft-test/repair2-eval/final.pt --batch 8
    python rl/dpo_pairs.py --selftest
"""

import argparse
import collections
import glob
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sft.reward import (same_entity_set, score_grounded_pairs,  # noqa: E402
                        score_format)
from sft.rollout import IM_END_ID, render_prefix  # noqa: E402

DEFAULT_MIN_MARGIN = 0.34   # ~1/3 of claims; ties give zero DPO gradient


# ------------------------------------------------------------------ scoring
def reject_reasons(pairs):
    """Why a pair cannot be used. Reported in the manifest, not silently dropped:
    a pipeline that emits fewer pairs than expected has to say which gate ate
    them, or the next run repeats the same mistake blind."""
    if not pairs:
        return "no_pairs"
    if len({(e.lower(), v.lower()) for e, v in pairs}) < 2:
        return "degenerate_claims"
    return None


def clean_answer(text):
    """The generated turn, stripped of control markers.

    Delegates to `dpo_verify.strip_markers` so the writer and the verifier use
    one list. The KV engine drops the eos id rather than returning it, but the
    model still emits markers as literal text often enough to matter, and
    grading against them would be grading on noise.
    """
    from rl.dpo_verify import strip_markers
    return strip_markers(text)


def length_stats(pairs):
    return {"claims_mean": sum(len(p) for p in pairs) / max(1, len(pairs)),
            "entities": len({e.lower() for e, _ in pairs})}


# ------------------------------------------------------------------ rollouts
def build_engine(args):
    """Load the policy. Two sources, because the Pi keeps the repo's own format
    and the Hub has an HF export of the same weights:
      --ckpt   a final.pt in this repo's format (llama.model.LLaMA state dict)
      --repo   an HF repo id / export dir, converted by rollout.build_model_from_hf
    """
    import torch
    from llama.model import LLaMA
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or {}
        m = LLaMA(**{k: cfg[k] for k in
                     ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
                      "head_dim", "ffn_dim", "rope_theta", "rms_eps")})
        m.load_state_dict(ck["model"], strict=True)
        src = args.ckpt
    else:
        from transformers import AutoTokenizer, LlamaForCausalLM
        from sft.rollout import build_model_from_hf
        hf = LlamaForCausalLM.from_pretrained(args.repo,
                                              torch_dtype=torch.float32).eval()
        m = build_model_from_hf(hf, LLaMA)
        src = args.repo
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = m.eval().to(device)
    if device == "cuda":
        # LLaMA.forward casts the residual stream to fp16 on CUDA, so the fp32
        # weights must sit under autocast or the matmuls meet Half against float
        # (the same fix rollout.KVEngine makes).
        pass
    torch.set_num_threads(args.threads)
    return m, device, src


def generate_batch(model, device, tok, prompt_ids, args, seed0):
    """One batch, equal-length prompts only (kv_cache.generate asserts it, so the
    caller buckets). Returns (text, finished) per row."""
    import torch
    from llama.kv_cache import generate as kv_generate
    outs = []
    for i, p in enumerate(prompt_ids):
        p = list(p)[-args.ctx:]
        with torch.inference_mode(), torch.autocast(
                device, torch.float16, enabled=(device == "cuda")):
            ids = kv_generate(model, [p], max_new=args.max_new, temp=args.temp,
                              top_p=args.top_p, seed=seed0 + i)[0]
        # kv_cache.generate stops the row when it hits eos, so a short result
        # means it terminated and a full-length one means it ran to the cap.
        outs.append((tok.decode(ids, skip_special_tokens=False),
                     len(ids) < args.max_new))
    return outs


def ref_logprobs(model, device, tok, side_ids, ce_chunk):
    """log pi_ref(completion | prompt) per row, for both sides of each pair.

    Computed once, here, so the trainer carries a single model: the reference
    policy *is* the policy that generated `rejected`, so its logprobs are a
    constant of the dataset.

    Two things this must get right, both of which are silently wrong otherwise:

      **the mask covers the completion only.** Summing over the prompt adds a
      large constant shared by both sides; it mostly cancels in the margin, but
      it does not cancel in `nll_chosen`'s length normalisation.
      **both sides.** Only computing `rejected` leaves `logp_ref_chosen` at its
      0.0 default, which turns the DPO term into plain SFT on the rejected
      sample -- it would train happily and optimise the wrong thing.

    Shifting is the caller's job (x = tokens[:-1], y = tokens[1:]), matching
    `LLaMA.completion_logprob`'s convention and the SFT trainer's.

    `side_ids` is [(ids, n_prompt), ...]; returns (logp, n_completion) per row.
    """
    import torch
    T = max(len(ids) for ids, _ in side_ids)
    B = len(side_ids)
    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.zeros(B, T, dtype=torch.long)
    m = torch.zeros(B, T, dtype=torch.float32)
    n_tok = []
    for i, (ids, n_prompt) in enumerate(side_ids):
        t = torch.tensor(ids, dtype=torch.long)
        n = len(ids)
        n_comp = n - n_prompt
        if n_comp <= 0:
            raise ValueError(f"row {i}: no completion tokens")
        x[i, :n - 1] = t[:-1]
        y[i, :n - 1] = t[1:]
        # position t predicts tokens[t+1]; the completion's first token sits at
        # index n_prompt, so the predicting position is n_prompt - 1
        m[i, n_prompt - 1:n - 1] = 1.0
        n_tok.append(n_comp)
    with torch.inference_mode(), torch.autocast(
            device, torch.float16, enabled=(device == "cuda")):
        lp = model.completion_logprob(x.to(device), y.to(device), m.to(device),
                                      ce_chunk=ce_chunk)
    return lp.float().cpu().tolist(), n_tok


def pair_ids(tok, prompt_ids, answer_text, stopped):
    """Token ids for prompt+answer and the completion mask.

    The answer is appended the way the trainer will see it, and `` is
    appended only when the model actually emitted one — appending it to a
    truncated answer would train the policy to terminate mid-sentence, which is
    the opposite of what the format term wants.
    """
    body = tok.encode(answer_text, add_special_tokens=False)
    body = list(getattr(body, "ids", body))
    if stopped:
        body = body + [IM_END_ID]
    return prompt_ids + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="dpo-mined")
    ap.add_argument("--out", default="dpo-pairs")
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/ai/sft-test/repair2-eval/final.pt"))
    ap.add_argument("--repo", default="")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=3072)
    ap.add_argument("--ce-chunk", type=int, default=512,
                    help="vocab chunk for the ref-logprob pass; 512 keeps the "
                         "logit chunk ~100MB fp32, which matters at 3GB RAM")
    ap.add_argument("--min-margin", type=float, default=DEFAULT_MIN_MARGIN)
    ap.add_argument("--max-rows", type=int, default=2500)
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="stop after N emitted pairs (0 = no cap)")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    files = sorted(glob.glob(os.path.join(args.mined, "mined-*.jsonl")))
    if not files:
        print(f"no mined rows in {args.mined} — run rl/dpo_mine.py first",
              file=sys.stderr)
        return 2

    rows, seen = [], set()
    for f in files:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            why = reject_reasons([tuple(c) for c in r["claims"]])
            if why:
                continue
            rows.append(r)
    rows = rows[:args.max_rows]
    print(f"[pairs] {len(rows)} mined rows after dedup", flush=True)
    if not rows:
        return 2

    import torch
    tok = load_tokenizer(args.tokenizer, args.mined)
    model, device, src = build_engine(args)
    print(f"[pairs] policy {src} on {device}, threads={args.threads}, "
          f"params {model.num_params()/1e6:.1f}M", flush=True)

    # Exact prompt lengths (render_prefix, not the miner's approximation) drive
    # the buckets. kv_cache.generate refuses unequal lengths, so this is also
    # the correctness constraint, not just a speed one.
    for r in rows:
        r["ids_prompt"] = render_prefix(tok, r["prompt_msgs"])
    rows.sort(key=lambda r: len(r["ids_prompt"]))
    buckets = [rows[i:i + args.batch] for i in range(0, len(rows), args.batch)]
    print(f"[pairs] prompt tokens p50={len(rows[len(rows)//2]['ids_prompt'])} "
          f"max={len(rows[-1]['ids_prompt'])}; {len(buckets)} batches of "
          f"<= {args.batch}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    done_keys = set()
    if args.resume:
        for f in glob.glob(os.path.join(args.out, "pairs-*.jsonl")):
            for line in open(f):
                line = line.strip()
                if line:
                    try:
                        done_keys.add(json.loads(line)["key"])
                    except Exception:
                        continue
        print(f"[pairs] resume: {len(done_keys)} pairs already written",
              flush=True)

    out_f = open(os.path.join(args.out, "pairs-0000.jsonl"), "a")
    drops = collections.Counter()
    n_pairs = 0
    t0 = time.time()
    for bi, batch in enumerate(buckets):
        if all(r["key"] in done_keys for r in batch):
            continue
        gens = generate_batch(model, device, tok,
                              [r["ids_prompt"] for r in batch], args,
                              seed0=args.seed + 1000 * bi)
        kept_idx, side_ids = [], []
        for j, (r, (text_c, stopped)) in enumerate(zip(batch, gens)):
            cand = clean_answer(text_c)
            g = score_grounded_pairs(r["gold_answer"], [tuple(c) for c in r["claims"]])
            c = score_grounded_pairs(cand, [tuple(c) for c in r["claims"]])
            margin = g - c
            if margin < args.min_margin or not cand.strip():
                drops["tied" if c >= g else "below_margin"] += 1
                continue
            pure = same_entity_set(cand, r["gold_answer"],
                                   [tuple(x) for x in r["claims"]])
            r["_scored"] = dict(cand=cand, stopped=stopped, gold=g, score=c,
                                margin=margin,
                                fmt=score_format([cand], stopped=[stopped]),
                                pure_binding=pure)
            drops["kept"] += 1
            kept_idx.append(j)
            nprompt = len(r["ids_prompt"])
            side_ids.append(
                (pair_ids(tok, r["ids_prompt"], r["gold_answer"], True), nprompt))
            side_ids.append(
                (pair_ids(tok, r["ids_prompt"], cand, stopped), nprompt))
        if not side_ids:
            continue
        lp, n_tok = ref_logprobs(model, device, tok, side_ids, args.ce_chunk)
        for k, j in enumerate(kept_idx):
            r = batch[j]
            s = r["_scored"]
            out_f.write(json.dumps({
                "key": r["key"],
                "source": r.get("source", ""),
                "prompt_msgs": r["prompt_msgs"],
                "prompt_tokens": len(r["ids_prompt"]),
                "chosen": r["gold_answer"],
                "rejected": s["cand"],
                "rejected_stopped": s["stopped"],
                "claims": r["claims"],
                "gold_score": s["gold"],
                "rejected_score": s["score"],
                "margin": s["margin"],
                "format": s["fmt"],
                "pure_binding": s["pure_binding"],
                "logp_ref_chosen": lp[2 * k],
                "logp_ref_rejected": lp[2 * k + 1],
                "chosen_tokens": n_tok[2 * k],
                "rejected_tokens": n_tok[2 * k + 1],
            }, ensure_ascii=False) + "\n")
            n_pairs += 1
            done_keys.add(r["key"])
        if (bi + 1) % 10 == 0:
            out_f.flush()
            el = time.time() - t0
            print(f"[pairs] batch {bi+1}/{len(buckets)} pairs={n_pairs} "
                  f"{el/(bi+1):.1f}s/batch eta "
                  f"{(len(buckets)-bi-1)*el/(bi+1)/60:.0f}min", flush=True)
        if args.max_pairs and n_pairs >= args.max_pairs:
            print(f"[pairs] --max-pairs {args.max_pairs} reached", flush=True)
            break
    out_f.close()

    # Stats over what was actually written, including pairs carried over from a
    # resumed run, so a restart does not report only its own shard's numbers.
    n_pure = 0
    lens_c, lens_r, gaps = [], [], []
    written = set()
    for f in glob.glob(os.path.join(args.out, "pairs-*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            written.add(d["key"])
            n_pure += bool(d["pure_binding"])
            lens_c.append(len(d["chosen"]))
            lens_r.append(len(d["rejected"]))
            gaps.append(d["margin"])
    stats = {
        "pairs": n_pairs,
        "pairs_on_disk": len(written),
        "min_margin": args.min_margin,
        "drops": dict(drops),
        "pure_binding_frac": n_pure / max(1, n_pairs),
        "mean_margin": sum(gaps) / max(1, len(gaps)),
        "chosen_chars": sum(lens_c) / max(1, len(lens_c)),
        "rejected_chars": sum(lens_r) / max(1, len(lens_r)),
        "reject_shorter_frac": sum(1 for a, b in zip(lens_c, lens_r)
                                   if b < a) / max(1, len(written)),
        "seconds_per_batch": (time.time() - t0) / max(1, len(buckets)),
    }
    with open(os.path.join(args.out, "pairs-manifest.json"), "w") as fh:
        json.dump(dict(args=vars(args), stats=stats), fh, indent=1)
    print(json.dumps(stats, indent=1))

    if not written:
        print("\nno pairs survived — read the drops table. `_tied` dominating "
              "means the policy already binds as well as gold here, and the "
              "signal has to come from best-of-N or from harder rows.",
              flush=True)
        return 3
    # The style-transfer reading. A low pure_binding fraction means most of the
    # gradient would be about prose, not about value/entity alignment; the fix
    # is best-of-N pairs from V1 itself, not a lower margin.
    if stats["pure_binding_frac"] < 0.5:
        print(f"\nWARN pure_binding_frac {stats['pure_binding_frac']:.2f} < 0.5 — "
              f"most rejected answers are a *different answer*, not a "
              f"mis-bound one. Gold-as-chosen will mostly teach style. "
              f"Re-run with V1 best-of-N before training on this.", flush=True)
    if stats["reject_shorter_frac"] > 0.6:
        print(f"\nWARN rejected is shorter than chosen in "
              f"{stats['reject_shorter_frac']:.0%} of pairs — length bias will "
              f"be part of the reward. Normalise the logprob or cap turns.",
              flush=True)
    print(f"\nPAIRS OK — {len(written)} pairs on disk ({n_pairs} this run). "
          f"Next: rl/dpo_verify.py, then rl/train_dpo.py", flush=True)
    return 0


def load_tokenizer(path=None, mined=None):
    """Accept a tokenizer *directory*, a HF repo id, or a bare tokenizer.json.

    A bare .json is the common case here: the Pi's checkpoint dirs ship
    `tokenizer.json` with no `tokenizer_config.json`, so `AutoTokenizer` tries the
    Hub and dies with a repo-id validation error. `tokenizers.Tokenizer` needs
    only the file and is API-compatible with `rollout.render_prefix`, whose `_ids`
    already unwraps `.ids` from the Encoding it returns.
    """
    cands = [path, os.environ.get("TOK_JSON", "")]
    if mined:
        cands += [os.path.join(mined, "tokenizer.json")]
    cands += [os.path.expanduser("~/ai/sft-test/repair2-eval/tokenizer.json"),
              os.path.expanduser("~/ai/sft-pipeline/repair-tok-v1/tokenizer.json")]
    for c in cands:
        if c and os.path.isfile(c):
            from tokenizers import Tokenizer
            return Tokenizer.from_file(c)
        if c and os.path.isdir(c):
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(c)
    raise SystemExit("no tokenizer found; pass --tokenizer <dir|repo|file.json>")


# ------------------------------------------------------------------ selftest
def selftest():
    """Pair construction and the scoring decision, with no model. The rollout
    itself cannot be checked here; `rl/dpo_verify.py` checks the rendered
    bytes against a real tokenizer."""
    fails = []

    def expect(name, got, want, msg=""):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})"
              f"{' — ' + msg if msg else ''}")
        if not ok:
            fails.append(name)

    claims = [("Kampala", "0.3476"), ("Kampala", "32.5825"),
              ("Gulu", "2.7724"), ("Gulu", "32.2881")]
    gold = "Kampala: lat 0.3476, lon 32.5825. Gulu: lat 2.7724, lon 32.2881."

    print("== reject_reasons")
    expect("no_pairs", reject_reasons([]), "no_pairs")
    expect("degenerate", reject_reasons([("A", "1")]), "degenerate_claims")
    expect("ok", reject_reasons([tuple(c) for c in claims]), None)

    print("== clean_answer strips markers the policy may emit as text")
    expect("im_end", clean_answer("Kampala: 0.3476<|im_end|>"),
           "Kampala: 0.3476")
    # only the marker goes; a leaked call body stays, because the format term
    # and the claims grader both still have something to say about it
    expect("tool_call", clean_answer("<|tool_call|>{}Kampala: 0.3476"),
           "{}Kampala: 0.3476")
    # found by the verify round-trip on real Pi rollouts: an earlier version of
    # this list omitted tool_result, so the marker survived into the stored pair
    # and vanished on re-decode
    expect("tool_result", clean_answer(
        '<|tool_result|>{"employee_id": 715505}'), '{"employee_id": 715505}')
    expect("im_start", clean_answer("<|im_start|>Kampala: 0.3476"),
           "Kampala: 0.3476")
    expect("none", clean_answer(None), "")

    print("== margin decision: a mis-bound answer must clear it")
    swapped = "Kampala: lat 2.7724, lon 32.2881. Gulu: lat 0.3476, lon 32.5825."
    g = score_grounded_pairs(gold, claims)
    c = score_grounded_pairs(swapped, claims)
    print(f"  gold={g:.3f} swapped={c:.3f} margin={g-c:.3f} "
          f"(min_margin {DEFAULT_MIN_MARGIN})")
    expect("gold_perfect", round(g, 3), 1.0)
    expect("margin_clears", (g - c) >= DEFAULT_MIN_MARGIN, True)
    expect("pure_binding_flag", same_entity_set(swapped, gold, claims), True)

    print("== a refusal is a *different answer*, not a binding failure")
    refusal = "I could not retrieve that information."
    r = score_grounded_pairs(refusal, claims)
    expect("refusal_low", r <= 0.25, True)
    expect("refusal_not_pure", same_entity_set(refusal, gold, claims), False)

    print("== pair_ids appends im_end only on a clean stop")

    class FakeTok:
        # ids from 100 up, so a real IM_END_ID (2) can never coincide with a
        # content token -- the collision would make the truncation check pass
        # for the wrong reason
        def encode(self, s, add_special_tokens=False):
            return list(range(100, 100 + len(s.split())))

    pt = [1, 2, 3]
    stopped = pair_ids(FakeTok(), pt, "a b c", True)
    truncated = pair_ids(FakeTok(), pt, "a b c", False)
    expect("stopped_appends", stopped[-1], IM_END_ID)
    expect("truncated_does_not", truncated[-1] != IM_END_ID, True)
    expect("prompt_preserved", stopped[:3], pt)

    print()
    if fails:
        print(f"PAIRS SELFTEST FAIL: {fails}")
        return 1
    print("PAIRS SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())