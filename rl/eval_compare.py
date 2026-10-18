#!/usr/bin/env python3
"""G4 comparison harness — RL_PLAN_2 §9/§6.1.

Runs agent_rollout on the held-out split for a list of Ollama models, ×2 seeds,
temperature 1.0, through the unified serving path — then prints one table. Used to
compare a GRPO round against Stage-A and the previous round under an *identical*
command, so the numbers are commensurable (the stored eval records do not pin a
temperature, so re-running the baseline is the only way to be sure).

    python rl/eval_compare.py --models tinyballs-dpo tinyballs-grpo4 tinyballs-grpo5 \
        --out-dir outputs/grpo-r5-output
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_rollout import behavior_metrics  # noqa: E402


def evaluate(model, seed0, tasks, temp, out_dir):
    out = os.path.join(out_dir, f"eval-{model}-s{seed0}.jsonl")
    if not os.path.exists(out):
        print(f"[eval] {model} seed{seed0} -> {out}", flush=True)
        subprocess.run([sys.executable, os.path.join(_HERE, "agent_rollout.py"),
                        "--model", model, "--holdout", "--tasks", str(tasks),
                        "--seed0", str(seed0), "--temp", str(temp),
                        "--out", out], check=True)
    else:
        print(f"[eval] {model} seed{seed0} cached", flush=True)
    return [json.loads(line) for line in open(out) if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 500])
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    eps = {m: [] for m in a.models}
    for m in a.models:
        for s in a.seeds:
            eps[m] += evaluate(m, s, a.tasks, a.temp, a.out_dir)

    cols = [("check_pass", "check_pass_rate"), ("terminate", "terminate_rate"),
            ("verify", "verify_with_check"), ("vbs", "verify_before_submit"),
            ("loop", "loop_rate"), ("steps", "mean_steps"),
            ("valid", "tool_valid"), ("rbe", "read_before_edit")]
    head = f"{'model':<18} {'n':>4} " + " ".join(f"{c:>9}" for c, _ in cols)
    print("\n" + head)
    print("-" * len(head))
    for m in a.models:
        met = behavior_metrics(eps[m])
        cells = " ".join(f"{met.get(k):>9}" for _, k in cols)
        print(f"{m:<18} {met['n']:>4} {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
