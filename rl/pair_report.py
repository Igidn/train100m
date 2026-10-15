#!/usr/bin/env python3
"""Review a Stage A pairs file before spending DPO quota (RL_PLAN §2A/§7).

Prints the things that decide whether a pair set is trainable, plus decoded
samples so the bytes can be eyeballed:

  * shape/gate pass rate (reuses `rl/dpo_verify.check_row`)
  * chosen action mix and rejected behavior mix (does the student even call a tool?)
  * the rejected-length tail (a policy that rambles is a real signal, but a
    mostly-`num_predict`-terminated set means the student never closes a turn)
  * a few full samples: prompt tail + chosen + rejected

    python rl/pair_report.py --pairs rl/pairs/tier0.jsonl --samples 6
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .build_pairs import HEADER, canonicalize
    from .dpo_verify import check_row
except ImportError:
    from build_pairs import HEADER, canonicalize
    from dpo_verify import check_row

# actions, in the order the oracle makes them
ORACLE_PLAN = ["bash", "str_replace_editor", "str_replace_editor",
               "bash", "submit"]


def _calls(text):
    c = canonicalize({"role": "assistant", "content": text[len(HEADER):]
                      if text.startswith(HEADER) else text})
    return [t["name"] for t in c["tool_calls"]], c["content"]


def _arg(command):
    m = re.search(r'"arguments"\s*:\s*\{[^}]*?"command"\s*:\s*"([^"]+)"', command)
    return m.group(1) if m else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--tokenizer", default=os.path.expanduser(
        "~/.cache/tinyballs-import/tokenizer.json"))
    ap.add_argument("--samples", type=int, default=6)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs) if l.strip()]
    if not rows:
        print("no rows")
        return 2

    tok = None
    if os.path.isfile(a.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)

    # --- gates -------------------------------------------------------------
    n_bad = 0
    bad_kinds = {}
    for r in rows:
        bad, _ = check_row(tok, r) if tok else ([], {})
        bad = [b for b in bad if b != "bad_ref_logprob"] if tok else []
        if bad:
            n_bad += 1
            for b in bad:
                bad_kinds[b] = bad_kinds.get(b, 0) + 1

    # --- action mix --------------------------------------------------------
    chosen_actions, rejected_first = {}, {}
    rejected_calls, rejected_none = 0, 0
    lens, stopped = [], {"clean": 0, "truncated": 0}
    no_signal = 0
    for r in rows:
        c_names, _ = _calls(r["chosen"])
        for n in c_names:
            chosen_actions[n] = chosen_actions.get(n, 0) + 1
        r_names, r_text = _calls(r["rejected"])
        if r_names:
            rejected_calls += 1
            rejected_first[r_names[0]] = rejected_first.get(r_names[0], 0) + 1
        else:
            rejected_none += 1
        lens.append(len(r["rejected"]))
        stopped["clean" if r.get("rejected_stopped", True) else "truncated"] += 1
        if r["chosen"].strip() == r["rejected"].strip():
            no_signal += 1

    turn_pos = {}
    for r in rows:
        turn_pos[r["turn_index"]] = turn_pos.get(r["turn_index"], 0) + 1

    print(f"pairs file      : {a.pairs}")
    print(f"pairs / tasks   : {len(rows)} / {len({r['source'].rsplit(':',1)[0] for r in rows})}")
    print(f"check_row pass  : {len(rows)-n_bad}/{len(rows)}"
          + (f"  (failures: {bad_kinds})" if n_bad else ""))
    print(f"identical pairs : {no_signal} (should be 0 — builder drops these)")
    print(f"turn positions  : {dict(sorted(turn_pos.items()))}")
    print()
    print(f"chosen actions  : {chosen_actions}")
    print(f"  (oracle plan per trajectory ≈ {ORACLE_PLAN})")
    print(f"rejected has call: {rejected_calls}/{len(rows)}"
          f"   no-call: {rejected_none}")
    print(f"rejected 1st call: {dict(sorted(rejected_first.items(), key=lambda kv: -kv[1]))}")
    print(f"rejected stopped : {stopped}  (truncated = hit num_predict, never closed the turn)")
    print(f"rejected chars   : mean={statistics.mean(lens):.0f} "
          f"median={statistics.median(lens):.0f} max={max(lens)}")

    # --- samples -----------------------------------------------------------
    if tok and a.samples:
        from sft.rollout import render_prefix
        print("\n" + "=" * 78)
        # spread the samples across turn positions
        picks = []
        by_pos = {}
        for r in rows:
            by_pos.setdefault(r["turn_index"], []).append(r)
        for pos in sorted(by_pos):
            picks.append(by_pos[pos][0])
            if len(picks) >= a.samples:
                break
        for r in picks:
            ids = render_prefix(tok, r["prompt_msgs"])
            tail = tok.decode(ids[-90:], skip_special_tokens=False)
            print(f"\n--- {r['key']}  (turn {r['turn_index']}, "
                  f"prompt {len(ids)} tok, stopped={r.get('rejected_stopped')})")
            print("  prompt tail : " + repr(tail))
            print("  CHOSEN      : " + repr(r["chosen"]))
            print("  rejected    : " + repr(r["rejected"][:400]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
