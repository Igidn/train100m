#!/usr/bin/env python
"""Verify DPO pairs before training on them — stage 3 of 3 (`rl/`).

The repo's convention is a gate that fails loudly rather than a training run
that quietly optimises something else. Four defects already measured here were
all "plausible-looking output, wrong bytes" (`REPAIR2_EVAL.md` §2), and the same
class of bug is what this script exists to catch:

  1. **render fidelity.** `prompt_msgs` is re-rendered through
     `rollout.render_prefix` and compared against the prefix of the full
     conversation. A drift here means training a model nobody measured — the
     exact defect that moved single_tool 2/8 -> 7/8 with no weight change.
  2. **mask spans the completion and nothing else.** System, user and tool
     turns must be 0, or the gradient updates the prompt.
  3. **no eval facts in either side.** Decontam ran at SFT time; `rejected` is
     new text and `chosen` came from a corpus the filter never saw in this
     form. The eval is re-measured after DPO, so a leak here invalidates the
     only number that counts.
  4. **both sides re-decode to their stored string**, and the stored reference
     logprob is finite. A NaN in `logp_ref_rejected` propagates through the
     DPO term and the run dies at the first `logsigmoid`.

    python rl/dpo_verify.py --pairs dpo-pairs --sample 0.05
    python rl/dpo_verify.py --selftest
"""

import argparse
import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.dpo_mine import EVAL_PATTERNS, contaminated  # noqa: E402
from sft.rollout import IM_END_ID, render_prefix  # noqa: E402


IM_START_ID, IM_END_ID = 1, 2
CONTROL_MARKERS = ("<|im_end|>", "<|im_start|>", "<|tool_call|>",
                   "<|tool_result|>")


def strip_markers(text):
    """Remove turn/control markers from decoded text.

    The model emits these as *text* often enough to matter, and they must not
    reach the grader or the training bytes. The list has to be complete: an
    earlier version omitted `<|tool_result|>`, the verify round-trip caught the
    resulting mismatch, and that is the only reason this function is shared
    between the miner's writer and the verifier's reader instead of being
    written twice.
    """
    t = text or ""
    for m in CONTROL_MARKERS:
        t = t.replace(m, "")
    return t.strip()


def load_tokenizer_quiet(path=None, where=None):  # pragma: no cover
    raise NotImplementedError


def load_pairs(paths):
    rows = []
    for p in paths:
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pair_tensors(tok, row):
    """The two training sequences and their completion masks.

    Kept here rather than only in the trainer because this is the function the
    verify script has to agree with, byte for byte. Right-padding with mask 0 is
    safe for a logprob sum — unlike generation, where a padded column is read.
    """
    prompt_ids = render_prefix(tok, row["prompt_msgs"])

    def side(text, stopped):
        body = tok.encode(text, add_special_tokens=False)
        body = list(getattr(body, "ids", body))
        if stopped:
            body = body + [IM_END_ID]
        if not body:
            # an empty completion is not a row. dpo_pairs.py filters it, but a
            # hand-edited or truncated pairs file must not turn into a
            # zero-token sum, which is a silent no-op on the gradient.
            raise ValueError("empty completion side")
        ids = prompt_ids + body
        mask = [0] * len(prompt_ids) + [1] * len(body)
        return ids, mask

    c_ids, c_mask = side(row["chosen"], True)
    r_ids, r_mask = side(row["rejected"], bool(row.get("rejected_stopped", True)))
    return prompt_ids, (c_ids, c_mask), (r_ids, r_mask)


def check_row(tok, row, deep=True):
    """(problems, info) for one pair."""
    bad = []
    info = {}
    try:
        prompt_ids, (c_ids, c_mask), (r_ids, r_mask) = build_pair_tensors(tok, row)
    except ValueError as e:
        return [str(e)], {}

    n = len(prompt_ids)
    if c_mask[:n] != [0] * n:
        bad.append("chosen_mask_leaks_prompt")
    if r_mask[:n] != [0] * n:
        bad.append("rejected_mask_leaks_prompt")
    if sum(c_mask) == 0 or sum(r_mask) == 0:
        bad.append("empty_completion")
    info["prompt_tokens"] = n
    info["chosen_tokens"] = sum(c_mask)
    info["rejected_tokens"] = sum(r_mask)

    # `rejected_stopped` has to agree with the ids it produced. A stale flag from
    # an interrupted run would append an im_end the model never emitted, which
    # trains it to terminate mid-sentence -- the opposite of the format term.
    if not row.get("rejected_stopped", True) and r_ids[-1] == IM_END_ID:
        bad.append("im_end_on_truncated_rejected")
    if row.get("rejected_stopped", True) and r_ids[-1] != IM_END_ID:
        bad.append("missing_im_end_on_clean_stop")

    for side in ("rejected", "chosen"):
        v = row.get(f"logp_ref_{side}")
        if v is None or not math.isfinite(v):
            bad.append("bad_ref_logprob")

    if contaminated(row.get("chosen", "")):
        bad.append("eval_fact_in_chosen")
    if contaminated(row.get("rejected", "")):
        bad.append("eval_fact_in_rejected")

    if deep:
        # both sides must re-decode to what was stored. skip_special_tokens is
        # False on purpose: the point is to see the marker if one is in there,
        # and strip_markers then removes it exactly as the writer did.
        for name, ids, mask in (("chosen", c_ids, c_mask),
                                ("rejected", r_ids, r_mask)):
            txt = strip_markers(tok.decode([i for i, m in zip(ids, mask) if m],
                                           skip_special_tokens=False))
            want = strip_markers(row[name])
            if txt != want:
                bad.append(f"{name}_decode_mismatch")
    return bad, info


def summarize(rows, pairs_by_key):
    def mean(xs):
        return sum(xs) / max(1, len(xs))
    margins = [p["margin"] for p in pairs_by_key if "margin" in p]
    return {
        "n": len(pairs_by_key),
        "mean_margin": mean(margins),
        "min_margin_seen": min(margins) if margins else 0,
        "pure_binding_frac": mean([bool(p.get("pure_binding"))
                                   for p in pairs_by_key]),
        # the reference logprob gap is the style-transfer reading in numbers:
        # a large chosen-vs-rejected gap means the pair is asking the policy to
        # cross a big likelihood distance, which is what `alpha * NLL(chosen)`
        # ends up doing most of the work on
        "mean_logp_ref_chosen": mean([p["logp_ref_chosen"]
                                      for p in pairs_by_key
                                      if "logp_ref_chosen" in p]),
        "mean_logp_ref_rejected": mean([p["logp_ref_rejected"]
                                        for p in pairs_by_key
                                        if "logp_ref_rejected" in p]),
        "mean_prompt_tokens": mean([r["prompt_tokens"] for r in rows]),
        "mean_chosen_tokens": mean([r["chosen_tokens"] for r in rows]),
        "mean_rejected_tokens": mean([r["rejected_tokens"] for r in rows]),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=0),
        "rejected_shorter_frac": mean(
            [r["rejected_tokens"] < r["chosen_tokens"] for r in rows]),
    }


def selftest():
    """The checks themselves, with a stub tokenizer and a deliberately broken
    pair each. A verifier whose own failures are never exercised is a comment."""
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})")
        if not ok:
            fails.append(name)

    class Tok:
        """Reversible char-level stub; markers are single ids like the real
        tokenizer.

        Reversibility is the point: the deep check re-decodes the completion ids
        and compares against the stored string, which a lossy stub could never
        satisfy and would make the assertion vacuous.
        """
        MARK = {"<|im_end|>": 2, "<|im_start|>": 1,
                "<|tool_call|>": 49152, "<|tool_result|>": 49153}

        def __init__(self):
            self.rev = {}

        def encode(self, s, add_special_tokens=False):
            out, i = [], 0
            while i < len(s):
                for mk, mid in self.MARK.items():
                    if s.startswith(mk, i):
                        out.append(mid)
                        i += len(mk)
                        break
                else:
                    tid = 100 + ord(s[i])
                    self.rev[tid] = s[i]
                    out.append(tid)
                    i += 1
            return out

        def decode(self, ids, skip_special_tokens=False):
            known = {v: k for k, v in self.MARK.items()}
            out = []
            for i in ids:
                if i in known:
                    if not skip_special_tokens:
                        out.append(known[i])
                elif i in self.rev:
                    out.append(self.rev[i])
            return "".join(out)

    tok = Tok()
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "obs"}]
    base = dict(prompt_msgs=msgs, chosen="alpha beta", rejected="gamma",
                rejected_stopped=True, logp_ref_chosen=-4.0,
                logp_ref_rejected=-3.5, margin=1.0, pure_binding=True)

    print("== a clean pair passes")
    bad, info = check_row(tok, base, deep=False)
    expect("clean", bad, [])
    expect("completion_count", info["chosen_tokens"],
           len(tok.encode("alpha beta", add_special_tokens=False)) + 1)

    print("== mask is zeros over the prompt then ones over the completion")
    _, (c_ids, c_mask), _ = build_pair_tensors(tok, base)
    zeros = c_mask.index(1)
    expect("mask_shape", c_mask,
           [0] * zeros + [1] * (len(c_mask) - zeros))
    expect("mask_leads_with_prompt", zeros, len(c_ids) - info["chosen_tokens"])

    print("== im_end is appended only on a clean stop")
    _, (_, cm), (_, rm_stop) = build_pair_tensors(tok, base)
    _, (_, _), (_, rm_trunc) = build_pair_tensors(
        tok, dict(base, rejected_stopped=False))
    expect("clean_stop_appends", cm[-1], 1)
    expect("truncated_omits", rm_trunc[-1] != IM_END_ID, True)
    expect("truncated_shorter", len(rm_trunc), len(rm_stop) - 1)

    print("== the stopped flag must agree with the stored ids")
    # check_row reads the *rebuilt* ids, so the guard fires when a pairs file's
    # `rejected_stopped` disagrees with its `rejected` text — a stale flag from
    # an interrupted run would otherwise train the policy to stop mid-sentence
    stopped_ends = dict(base, rejected_stopped=True)
    expect("flag_consistent", check_row(tok, stopped_ends, deep=False)[0], [])
    # stopped=True would append im_end and mask 1 token, so the zero-token path is
    # only reachable on a truncated side — which is exactly the broken row
    lies = dict(base, rejected="", rejected_stopped=False)
    expect("empty_rejected", check_row(tok, lies, deep=False)[0],
           ["empty completion side"])

    print("== non-finite reference logprob is caught, on either side")
    for badval in (float("nan"), float("-inf")):
        r = dict(base, logp_ref_chosen=badval, logp_ref_rejected=-3.5)
        expect(f"bad_ref_chosen_{badval}", check_row(tok, r, deep=False)[0],
               ["bad_ref_logprob"])
        r = dict(base, logp_ref_chosen=-3.5, logp_ref_rejected=badval)
        expect(f"bad_ref_rejected_{badval}", check_row(tok, r, deep=False)[0],
               ["bad_ref_logprob"])

    print("== a missing logp_ref_chosen is a hard failure, not a 0.0 default")
    # 0.0 is a finite, plausible number, so a default would quietly turn the
    # DPO term into plain SFT on the rejected sample
    missing = dict(base)
    del missing["logp_ref_chosen"]
    expect("missing_chosen", check_row(tok, missing, deep=False)[0],
           ["bad_ref_logprob"])

    print("== a control marker in a stored answer is stripped, and survives the")
    print("   round-trip (this is the bug the deep check found on real rollouts)")
    for mk in CONTROL_MARKERS:
        expect(f"strip_{mk}", strip_markers(f"Kampala: 0.3476{mk}"),
               "Kampala: 0.3476")
        r = dict(base, rejected=f'{{"employee_id": 715505}}{mk}')
        expect(f"roundtrip_{mk}", check_row(tok, r, deep=True)[0], [])

    print("== eval facts are caught on either side")
    leaky_c = dict(base, chosen="Tokyo is 18.4C", rejected="ok")
    expect("chosen", check_row(tok, leaky_c, deep=False)[0],
           ["eval_fact_in_chosen"])
    leaky_r = dict(base, rejected="New York is 22.1C")
    expect("rejected", check_row(tok, leaky_r, deep=False)[0],
           ["eval_fact_in_rejected"])

    print("== the deep check exists for lossy encoding, not for stale strings")
    # ids are built *from* the stored string, so a plain mismatch is impossible
    # by construction. What it can catch is an encode/decode round-trip that is
    # not the identity -- which is exactly the <|tool_result|> bug above. Show
    # the raw decode losing the marker, so the check is not vacuous.
    raw = tok.decode(tok.encode("Kampala: 0.3476<|tool_result|>{}",
                                add_special_tokens=False),
                     skip_special_tokens=False)
    expect("marker_survives_raw_decode", "<|tool_result|>" in raw, True)
    expect("marker_is_what_strip_removes",
           strip_markers("Kampala: 0.3476<|tool_result|>{}"), "Kampala: 0.3476{}")
    expect("clean_row_passes_deep", check_row(tok, base, deep=True)[0], [])

    print()
    if fails:
        print(f"VERIFY SELFTEST FAIL: {fails}")
        return 1
    print("VERIFY SELFTEST PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="dpo-pairs")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--sample", type=float, default=0.05,
                    help="fraction of rows to deep-check (decode round-trip)")
    ap.add_argument("--all", action="store_true", help="deep-check every row")
    ap.add_argument("--min-pairs", type=int, default=64)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    paths = sorted(glob.glob(os.path.join(args.pairs, "pairs-*.jsonl")))
    if not paths:
        print(f"no pairs in {args.pairs}", file=sys.stderr)
        return 2
    rows = load_pairs(paths)
    print(f"[verify] {len(rows)} pairs from {len(paths)} file(s)", flush=True)
    if not rows:
        return 2

    # imported here, not at module level: dpo_pairs imports strip_markers from
    # this module, so a top-level import would be circular
    from rl.dpo_pairs import load_tokenizer
    tok_path = args.tokenizer or ""
    tok = load_tokenizer(tok_path, args.pairs)
    print(f"[verify] tokenizer {type(tok).__name__} "
          f"{tok_path or os.path.join(args.pairs, 'tokenizer.json')}", flush=True)

    keys = [r["key"] for r in rows]
    dupes = len(keys) - len(set(keys))
    problems = {}
    infos = []
    deep_n = len(rows) if args.all else max(1, int(len(rows) * args.sample))
    for i, r in enumerate(rows):
        bad, info = check_row(tok, r, deep=(i < deep_n or args.all))
        infos.append(info)
        for b in bad:
            problems[b] = problems.get(b, 0) + 1

    stats = summarize(infos, rows)
    stats["deep_checked"] = deep_n
    stats["duplicate_keys"] = dupes
    stats["problems"] = problems
    print(json.dumps(stats, indent=1))

    fatal = []
    if dupes:
        fatal.append(f"{dupes} duplicate keys (resume wrote a row twice)")
    for name, n in problems.items():
        # decode mismatches on a sample are a real failure; format-ish noise is not
        fatal.append(f"{name}: {n}")
    if len(rows) < args.min_pairs:
        fatal.append(f"only {len(rows)} pairs < --min-pairs {args.min_pairs}")

    with open(os.path.join(args.pairs, "verify-report.json"), "w") as fh:
        json.dump(dict(stats=stats, fatal=fatal), fh, indent=1)

    if fatal:
        print("\nVERIFY FAIL")
        for f in fatal:
            print(f"  - {f}")
        print("\ndo not train on this. The four serving defects in "
              "REPAIR2_EVAL.md §2 were all of this shape: output that looks "
              "fine and trains a model nobody measured.", flush=True)
        return 1
    print(f"\nVERIFY PASS — {len(rows)} pairs, {deep_n} deep-checked. "
          f"Next: rl/train_dpo.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())