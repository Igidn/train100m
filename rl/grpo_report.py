#!/usr/bin/env python3
"""Gate G0 — reward variance of a sample pool, before any GPU time — RL_PLAN_2 §9.

RL plan 2 §9: "G0 — reward variance. Fraction of groups with non-zero reward
spread at G=16. If it is ~0, the reward is too sparse and no algorithm will
learn. Tune the §4.2 ladder until it is alive. *This is the gate that decides
whether the plan is real.*"

This reads `rl/sample_pool.py` output and prints, per group and in aggregate,
exactly the things that decide it: the reward spread, the degeneracy/drop rate,
the tier mix (which rung of the §4.2 ladder is carrying the signal), the
anti-hack counts, and the action mix. `--train-log` reads a `grpo-state.json`
instead and prints the KL / advantage-magnitude trend of a training run.

    python rl/grpo_report.py --samples rl/samples/batch-0001.jsonl
    python rl/grpo_report.py --train-log /kaggle/working/grpo-state.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# G0 thresholds. A G=16 group at a 12% pass rate is often all-fail; the ladder is
# supposed to move that. These are *starting* readings (RL_PLAN_2 §4.2/§14), not a
# law — the decision is "is there spread to learn from", not the exact cut.
PASS_SPREAD = 0.20
MARGINAL_SPREAD = 0.05


def load_rows(paths):
    """All rows from one or more jsonl files (globs allowed)."""
    rows = []
    for pat in paths:
        files = sorted(glob.glob(pat)) if any(c in pat for c in "*?[") else [pat]
        for p in files:
            if not os.path.exists(p):
                print(f"  (missing: {p})", file=sys.stderr)
                continue
            for line in open(p):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def report_samples(paths) -> int:
    rows = load_rows(paths)
    if not rows:
        print("no rows", file=sys.stderr)
        return 2
    # a *group* is the G episodes of one task; an *episode* is one rollout. The
    # spread/degeneracy that G0 reads is group-level; the tier mix is episode-level.
    groups, episodes = {}, {}
    for r in rows:
        groups.setdefault(r.get("group_id"), r)
        episodes.setdefault(r.get("episode_id"), r)
    n_groups, n_eps = len(groups), len(episodes)
    degen = [g for g, r in groups.items() if r.get("degenerate")]
    spreads = [r.get("group_std", 0.0) for r in groups.values()]
    means = [r.get("group_mean", 0.0) for r in groups.values()]
    pass_eps = [e for e, r in episodes.items() if r.get("reward") == 1.0]
    sub_eps = [e for e, r in episodes.items() if r.get("submitted")]
    ver_eps = [e for e, r in episodes.items() if r.get("verified")]
    tiers = Counter(r.get("tier", "?") for r in episodes.values())
    anti = Counter(a for r in episodes.values() for a in (r.get("anti_hack") or []))
    rewards = Counter(round(r.get("reward", -1), 3) for r in episodes.values())
    turns = Counter(r.get("turn_index", -1) for r in rows)
    advs = [r.get("advantage", 0.0) for r in episodes.values()
            if not r.get("degenerate")]

    # action mix, read off the completions
    actions = Counter()
    try:
        from build_pairs import HEADER, canonicalize
        for r in rows:
            content = (r.get("completion") or "")
            if content.startswith(HEADER):
                content = content[len(HEADER):]
            for tc in canonicalize({"role": "assistant", "content": content})["tool_calls"]:
                actions[tc["name"]] += 1
    except Exception as e:  # noqa: BLE001
        print(f"  (action mix unavailable: {e})", file=sys.stderr)

    spread_frac = (sum(1 for s in spreads if s > 0) / n_groups) if n_groups else 0.0
    print(f"samples        : {', '.join(paths)}")
    print(f"groups/rows    : {n_groups} tasks / {n_eps} episodes / {len(rows)} rows")
    print(f"degenerate     : {len(degen)}/{n_groups} dropped "
          f"({len(degen)/max(1,n_groups):.1%})")
    print(f"reward spread  : {spread_frac:.1%} of groups have std>0   "
          f"<- GATE G0")
    if spreads:
        print(f"group std      : mean {statistics.fmean(spreads):.3f}  "
              f"max {max(spreads):.3f}")
        print(f"group mean     : mean {statistics.fmean(means):.3f}  "
              f"episode pass-rate {len(pass_eps)/max(1,n_eps):.1%}")
    if advs:
        print(f"|advantage|    : mean {statistics.fmean(abs(a) for a in advs):.3f}  "
              f"max {max(abs(a) for a in advs):.3f}  (non-degenerate eps only)")
    print(f"tier mix (eps) : {dict(tiers.most_common())}")
    print(f"reward (eps)   : {dict(sorted(rewards.items()))}")
    print(f"submit rate    : {len(sub_eps)/max(1,n_eps):.1%} of episodes ended on "
          f"submit (§4.2b — was the termination reward alive?)")
    print(f"verify rate    : {len(ver_eps)/max(1,n_eps):.1%} of episodes re-ran the "
          f"check after their last edit (§4.2b — submit *and* verify bank the reward)")
    print(f"anti-hack (eps): {dict(anti) or '{}'}")
    print(f"turns/row group: {dict(sorted(turns.items()))}")
    if actions:
        print(f"action mix     : {dict(actions.most_common())}")

    verdict = ("PASS" if spread_frac >= PASS_SPREAD else
               "MARGINAL" if spread_frac >= MARGINAL_SPREAD else "DEAD")
    print(f"\nG0 verdict: {verdict} — {spread_frac:.1%} live groups "
          f"(pass>={PASS_SPREAD:.0%}, marginal>={MARGINAL_SPREAD:.0%})")
    if verdict == "DEAD":
        print("  The ladder is not creating spread. Per §12, the honest fallback "
              "is more tasks and a coarser/better ladder — not a bigger model.")
    return 0


def report_train_log(paths) -> int:
    """KL / advantage trend from a `grpo-state.json` `hist` list (train step log)."""
    hist = []
    for p in paths:
        if os.path.exists(p):
            try:
                obj = json.load(open(p))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict) and isinstance(obj.get("hist"), list):
                hist = obj["hist"]
                print(f"train log      : {p}  ({len(hist)} steps)")
                break
    if not hist:
        print("no hist in the given files", file=sys.stderr)
        return 2
    keys = ["loss", "pg", "kl", "mean_logp", "adv_abs_mean", "n_rows", "skipped"]
    print(f"{'step':>5} " + " ".join(f"{k:>12}" for k in keys))
    picks = hist[::max(1, len(hist) // 20)]
    if picks[-1] is not hist[-1]:
        picks = picks + [hist[-1]]
    for s in picks:
        print(f"{s.get('step', '?'):>5} "
              + " ".join(f"{float(s.get(k, 0)):>12.4f}" for k in keys))
    # the §12 failure signature: KL rising while the reward/proxy is flat
    kls = [float(s.get("kl", 0)) for s in hist]
    if len(kls) >= 4:
        half = len(kls) // 2
        early, late = statistics.fmean(kls[:half]), statistics.fmean(kls[half:])
        print(f"\nKL early {early:.4f} -> late {late:.4f}  "
              f"({'RISING — watch for collapse (RL_PLAN_2 §12)' if late > 2*early + 1e-4 else 'stable'})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="*", default=[], help="sample_pool jsonl file(s)")
    ap.add_argument("--train-log", nargs="*", default=[], help="grpo-state.json file(s)")
    a = ap.parse_args()
    if a.train_log:
        return report_train_log(a.train_log)
    return report_samples(a.samples or ["rl/samples/*.jsonl"])


if __name__ == "__main__":
    raise SystemExit(main())
