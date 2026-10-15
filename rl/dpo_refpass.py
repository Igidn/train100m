#!/usr/bin/env python
"""Fill `logp_ref_chosen`/`logp_ref_rejected` on a Stage A pairs file (RL_PLAN §2A).

`rl/build_pairs.py` runs on the Mac with no torch, so it leaves the two reference
logprobs null. The DPO trainer treats `log pi_ref` as a *constant of the dataset*
(the reference policy is the policy that generated `rejected`), so this pass runs
once, on the torch box, as step 0 before `rl/train_dpo.py`:

    python rl/dpo_refpass.py --pairs pairs/tier0.jsonl \\
        --out pairs/pairs-tier0.jsonl --tokenizer pairs/tokenizer.json \\
        --base-repo igidn/tinyballs-v1 --base-file REPAIR2/final.pt

The invariant that makes this correct: **the checkpoint here is the same one
`rejected` came from**, so `logp_ref_*` are the initial policy's logprobs. If it
is not, DPO measures the margin against the wrong origin and still "trains".

Sequence construction is shared with the trainer byte-for-byte via
`rl.dpo_verify.build_pair_tensors` (prompt + completion, completion-only mask,
im_end appended only when the model stopped) — so a ref computed here is the ref
the trainer scores. Refs are computed on the *untruncated* sequence, matching the
`MAX_LEN` note in `build_pair_batch`.

    python rl/dpo_refpass.py --dry --pairs ... # no torch model; checks shapes
    python rl/dpo_refpass.py --selftest        # --dry on a synthetic pair
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

ARCH_KEYS = ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
             "head_dim", "ffn_dim", "rope_theta", "rms_eps")

IM_START = "<" + "|im_start|>"
HEADER = IM_START + "assistant\n"


def ref_logprobs(model, device, tok, side_ids, ce_chunk):
    """log pi_ref(completion | prompt) per row — the DPO primitive.

    Mirrors `rl/dpo_pairs.ref_logprobs` exactly (kept here so this script imports
    nothing from the reward machinery). Shift convention is the trainer's:
    x = tokens[:-1], y = tokens[1:], mask aligned so only completion tokens
    contribute. `side_ids` is [(ids, n_prompt), ...]; returns a float per row.
    """
    import torch
    T = max(len(ids) for ids, _ in side_ids)
    B = len(side_ids)
    x = torch.zeros(B, T, dtype=torch.long)
    y = torch.zeros(B, T, dtype=torch.long)
    m = torch.zeros(B, T, dtype=torch.float32)
    for i, (ids, n_prompt) in enumerate(side_ids):
        t = torch.tensor(ids, dtype=torch.long)
        n = len(ids)
        if n - n_prompt <= 0:
            raise ValueError(f"row {i}: no completion tokens")
        x[i, :n - 1] = t[:-1]
        y[i, :n - 1] = t[1:]
        m[i, n_prompt - 1:n - 1] = 1.0
    with torch.inference_mode(), torch.autocast(
            device, torch.float16, enabled=(device == "cuda")):
        lp = model.completion_logprob(x.to(device), y.to(device), m.to(device),
                                      ce_chunk=ce_chunk)
    return lp.float().cpu().tolist()


def load_tok(path):
    """A transformers tokenizer from either a directory or a bare tokenizer.json.

    `AutoTokenizer.from_pretrained` only accepts a *directory* (or repo id), not a
    `tokenizer.json` file — so a file path falls back to a fast tokenizer built
    from the same file, which yields identical ids.
    """
    from transformers import AutoTokenizer
    if os.path.isdir(path):
        return AutoTokenizer.from_pretrained(path)
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(tokenizer_file=path)


# --------------------------------------------------------------------------- model
def load_model(args, device):
    import torch
    from llama.model import LLaMA
    path = args.base_ckpt_path
    if not path or not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        dest = args.base_dir
        os.makedirs(dest, exist_ok=True)
        path = hf_hub_download(repo_id=args.base_repo, filename=args.base_file,
                               token=token or None, local_dir=dest)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    arch = {k: cfg.get(k) for k in ARCH_KEYS}
    if any(v is None for v in arch.values()):
        raise SystemExit(f"checkpoint {path} is missing cfg keys: {cfg}")
    model = LLaMA(**arch)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    model.eval().to(device)
    return model, path


# --------------------------------------------------------------------------- run
def sides_for(tok, row):
    """(prompt_ids, chosen_side, rejected_side) via the trainer's own builder."""
    from dpo_verify import build_pair_tensors
    prompt_ids, (c_ids, _), (r_ids, _) = build_pair_tensors(tok, row)
    n = len(prompt_ids)
    return prompt_ids, (c_ids, n), (r_ids, n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="", help="input pairs .jsonl")
    ap.add_argument("--out", default="", help="output .jsonl (default: in place)")
    ap.add_argument("--tokenizer", default="", help="tokenizer.json (default: next to --pairs)")
    ap.add_argument("--base-repo", default=os.environ.get("BASE_REPO", "igidn/tinyballs-v1"))
    ap.add_argument("--base-file", default=os.environ.get("BASE_CKPT_FILE", "REPAIR2/final.pt"))
    ap.add_argument("--base-ckpt-path", default=os.environ.get("BASE_CKPT_PATH", ""))
    ap.add_argument("--base-dir", default=os.environ.get("BASE_DIR", "/kaggle/working/base"))
    ap.add_argument("--ce-chunk", type=int, default=int(os.environ.get("CE_CHUNK", "2048")))
    ap.add_argument("--device", default="")
    ap.add_argument("--dry", action="store_true", help="build sides only, no model")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.pairs:
        ap.error("--pairs is required")

    tok_path = a.tokenizer or os.path.join(os.path.dirname(a.pairs), "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"tokenizer not found: {tok_path}")
    tok = load_tok(tok_path)

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {a.pairs}")
    out_path = a.out or a.pairs

    if a.dry:
        n_long = 0
        for r in rows:
            pid, (c, n), (rr, _) = sides_for(tok, r)
            assert len(c) > n and len(rr) > n
            n_long = max(n_long, n)
        print(f"[dry] {len(rows)} rows ok; max prompt {n_long} tok "
              f"-> {out_path} untouched")
        return 0

    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, path = load_model(a, device)
    print(f"[ref] model  : {path}", flush=True)
    print(f"[ref] device : {device}  pairs {len(rows)}", flush=True)

    t0 = time.time()
    out = []
    for i, r in enumerate(rows):
        try:
            _, c, rr = sides_for(tok, r)
        except Exception as e:  # noqa: BLE001 — a malformed row must not kill the pass
            print(f"  ! row {i} ({r.get('key')}): {e}", flush=True)
            continue
        lp = ref_logprobs(model, device, tok, [c, rr], a.ce_chunk)
        r = dict(r)
        r["logp_ref_chosen"] = lp[0]
        r["logp_ref_rejected"] = lp[1]
        out.append(r)
        if (i + 1) % 50 == 0 or i + 1 == len(rows):
            dt = time.time() - t0
            print(f"  {i+1}/{len(rows)}  {dt:.0f}s  "
                  f"({(i+1)/max(dt,1e-9):.1f} rows/s)", flush=True)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    # the DPO style-transfer reading: a large chosen-vs-rejected ref gap means the
    # pair crosses a big likelihood distance under the initial policy.
    if out:
        def mean(k):
            return sum(r[k] for r in out) / len(out)
        gaps = [r["logp_ref_chosen"] - r["logp_ref_rejected"] for r in out]
        print(f"\n[ref] wrote {len(out)} rows -> {out_path}")
        print(f"[ref] mean logp_ref_chosen   : {mean('logp_ref_chosen'):+.3f}")
        print(f"[ref] mean logp_ref_rejected : {mean('logp_ref_rejected'):+.3f}")
        print(f"[ref] mean ref gap (c-r)     : {sum(gaps)/len(gaps):+.3f}  "
              f"min {min(gaps):+.3f}  max {max(gaps):+.3f}")
    return 0 if out else 2


def selftest() -> int:
    """--dry on a synthetic tier-0 pair (no model, no network)."""
    import tempfile
    sys.path.insert(0, _HERE)
    from build_pairs import HEADER as H, TOOL_CALL
    # pick any tokenizer we can find
    cand = [os.path.expanduser("~/.cache/tinyballs-import/tokenizer.json"),
            os.path.join(_ROOT, "toktest", "tokenizer.json")]
    tok_path = next((c for c in cand if os.path.exists(c)), "")
    if not tok_path:
        print("REF SELFTEST SKIP (no tokenizer found)")
        return 0
    header = H
    row = {
        "key": "selftest:2", "source": "selftest", "turn_index": 2,
        "prompt_msgs": [{"role": "system", "content": "S"},
                        {"role": "user", "content": "U"}],
        "chosen": header + TOOL_CALL + '{"name": "bash", "arguments": {"command": "ls"}}',
        "rejected": header + "I am not sure.",
        "rejected_stopped": True,
        "logp_ref_chosen": None, "logp_ref_rejected": None,
    }
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pairs.jsonl")
        json.dump(row, open(p, "w"))
        import shutil
        tokdir = os.path.join(d, "tok")
        os.makedirs(tokdir, exist_ok=True)
        shutil.copyfile(tok_path, os.path.join(tokdir, "tokenizer.json"))
        cfg = os.path.join(os.path.dirname(tok_path), "tokenizer_config.json")
        if os.path.exists(cfg):
            shutil.copyfile(cfg, os.path.join(tokdir, "tokenizer_config.json"))
        argv = sys.argv
        sys.argv = ["dpo_refpass", "--pairs", p, "--tokenizer", tokdir, "--dry"]
        try:
            rc = main()
        finally:
            sys.argv = argv
    print("REF SELFTEST", "PASS" if rc == 0 else "FAIL")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
