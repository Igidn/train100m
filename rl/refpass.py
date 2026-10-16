#!/usr/bin/env python3
"""Fill `logpi_ref` on a GRPO sample file — one frozen pass — RL_PLAN_2 §3/§7.

GRPO needs `log pi_ref(y|x)`, and per §3.1 the reference for a round is the
*round's starting policy*, fixed for the whole round. So this runs once, on the
torch box, before `rl/train_grpo.py`, exactly as `rl/dpo_refpass.py` runs before
`rl/train_dpo.py`. Each round is a fresh RLHF problem with a fresh anchor, so the
anchor is recomputed on each round's samples, not carried across rounds.

The sequence for a sampled completion is built byte-for-byte the way the trainer
will build it: `render_prefix(prompt_msgs) + encode(completion) + [im_end]` iff the
turn stopped cleanly (`completion_stopped`; a `num_predict`-cut turn must not get
an imagined im_end — the same `rejected_stopped` rule as Stage A). If the two ever
diverge, the KL term measures the wrong origin and still "trains".

    python rl/refpass.py --samples rl/samples/batch-0001.jsonl \\
        --base-ckpt-path outputs/rl-run-final/DPO/final.pt
    python rl/refpass.py --selftest        # --dry on a synthetic sample, no model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .dpo_refpass import load_model, load_tok, ref_logprobs
    from .dpo_verify import IM_END_ID
except ImportError:  # python rl/refpass.py
    from dpo_refpass import load_model, load_tok, ref_logprobs
    from dpo_verify import IM_END_ID
from sft.rollout import render_prefix  # noqa: E402  (_ROOT is on sys.path)

TOKENIZER_CANDIDATES = [
    os.path.expanduser("~/.cache/tinyballs-import/tokenizer.json"),
    os.path.join(_ROOT, "toktest", "tokenizer.json"),
]


def sample_sides(tok, row):
    """(ids, n_prompt) for one sampled completion — the trainer's own convention."""
    prompt_ids = render_prefix(tok, row["prompt_msgs"])
    body = tok.encode(row["completion"], add_special_tokens=False)
    body = list(getattr(body, "ids", body))
    if row.get("completion_stopped", True):
        body = body + [IM_END_ID]
    if not body:
        raise ValueError("empty completion")
    return prompt_ids + body, len(prompt_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="", help="input samples .jsonl")
    ap.add_argument("--out", default="", help="output .jsonl (default: in place)")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--base-repo", default=os.environ.get("BASE_REPO", "igidn/tinyballs-v1"))
    ap.add_argument("--base-file", default=os.environ.get("BASE_CKPT_FILE", "REPAIR2/final.pt"))
    ap.add_argument("--base-ckpt-path", default=os.environ.get("BASE_CKPT_PATH", ""))
    ap.add_argument("--base-dir", default=os.environ.get("BASE_DIR", "/kaggle/working/base"))
    ap.add_argument("--ce-chunk", type=int, default=int(os.environ.get("CE_CHUNK", "2048")))
    ap.add_argument("--device", default="")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.samples:
        ap.error("--samples is required")

    tok_path = a.tokenizer or next(
        (c for c in [os.path.join(os.path.dirname(a.samples), "tokenizer.json")]
         + TOKENIZER_CANDIDATES if os.path.exists(c)), "")
    if not tok_path:
        raise SystemExit("tokenizer not found; pass --tokenizer")
    tok = load_tok(tok_path)

    rows = [json.loads(l) for l in open(a.samples) if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {a.samples}")
    out_path = a.out or a.samples

    if a.dry:
        n_long = 0
        for r in rows:
            ids, n = sample_sides(tok, r)
            assert len(ids) > n
            n_long = max(n_long, n)
        print(f"[dry] {len(rows)} rows ok; max prompt {n_long} tok -> {out_path} untouched")
        return 0

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, path = load_model(a, device)
    print(f"[ref] model  : {path}", flush=True)
    print(f"[ref] device : {device}  rows {len(rows)}", flush=True)

    t0 = time.time()
    out, skipped = [], 0
    for i, r in enumerate(rows):
        try:
            ids, n = sample_sides(tok, r)
        except Exception as e:  # noqa: BLE001 — one bad row must not kill the pass
            print(f"  ! row {i} ({r.get('key')}): {e}", flush=True)
            skipped += 1
            continue
        lp = ref_logprobs(model, device, tok, [(ids, n)], a.ce_chunk)
        r = dict(r)
        r["logpi_ref"] = lp[0]
        out.append(r)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            dt = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {dt:.0f}s  ({(i+1)/max(dt,1e-9):.1f} rows/s)",
                  flush=True)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    if out:
        lps = [r["logpi_ref"] for r in out]
        # per-*token* is the honest scale: rows are whole turns, so a raw sum
        # tracks length. The report only needs the sign and the spread.
        print(f"\n[ref] wrote {len(out)} rows -> {out_path} ({skipped} skipped)")
        print(f"[ref] logpi_ref: mean {sum(lps)/len(lps):+.3f}  "
              f"min {min(lps):+.3f}  max {max(lps):+.3f}")
    return 0 if out else 2


def selftest() -> int:
    """--dry on a synthetic sample row, no model and no network."""
    import tempfile
    sys.path.insert(0, _HERE)
    from build_pairs import HEADER, TOOL_CALL
    tok_path = next((c for c in TOKENIZER_CANDIDATES if os.path.exists(c)), "")
    if not tok_path:
        print("REF SELFTEST SKIP (no tokenizer found)")
        return 0
    import shutil
    row = {
        "key": "selftest:g0:t1", "group_id": "selftest:g0", "episode_id": "selftest:g0",
        "task_id": "selftest", "group_index": 0, "turn_index": 1,
        "prompt_msgs": [{"role": "system", "content": "S"},
                        {"role": "user", "content": "U"}],
        "completion": HEADER + TOOL_CALL + '{"name": "bash", "arguments": {"command": "ls"}}',
        "completion_stopped": True, "reward": 1.0, "advantage": 0.5,
        "logpi_ref": None,
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "samples.jsonl")
        json.dump(row, open(p, "w"))
        tokdir = os.path.join(d, "tok")
        os.makedirs(tokdir, exist_ok=True)
        shutil.copyfile(tok_path, os.path.join(tokdir, "tokenizer.json"))
        argv = sys.argv
        sys.argv = ["refpass", "--samples", p, "--tokenizer", tokdir, "--dry"]
        try:
            rc = main()
        finally:
            sys.argv = argv
    print("REF SELFTEST", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
