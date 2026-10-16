#!/usr/bin/env python3
"""Sample pool: G graded episodes per task -> the GRPO row shape — RL_PLAN_2 §3/§7.

Per task it runs **G = 16 full episodes at temperature 1.0** through the unified
serving path (`agent_env.run_episode`, `agent_rollout.OllamaPolicy`), grades each
with `rl/verifier.py`, and folds the group into rows the trainer consumes. The
Mac generates; nothing here needs torch (RL_PLAN_2 §5.1/§10).

Three things this file gets right, each a §3.1 trap:

  * **Group advantage is computed here, once, and stored.** At train time the
    anchor must not move, and a group must never be split across ranks with a
    half-group mean. Advantages are a property of the offline sample file, so
    "whole groups on a rank" is automatic and free.
  * **Zero-variance groups are kept but flagged.** A G=16 group at ~12% pass is
    usually all-fail; those are `degenerate`, carry `advantage=0`, and the trainer
    drops them. They are written anyway so `rl/grpo_report.py` — Gate G0 — can
    measure the drop rate before a single GPU-second is spent.
  * **Turn-level rows, episode-level advantage.** `log pi(episode)` is the sum of
    per-turn logprobs over turns whose mask is the turn only; the tool
    observations between turns are context, never loss. Every row of an episode
    shares that episode's advantage, so the sum over rows recovers the episode.

    python rl/sample_pool.py --selftest                       # Gate G0, no model
    python rl/sample_pool.py --tasks 100 --group 16 --out rl/samples/batch-0001.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .agent_env import HOLDOUT_SEED0, generate_task, run_episode
    from .agent_rollout import OllamaPolicy, control_policy, oracle_policy
    from .build_pairs import HEADER, body
    from .verifier import grade
except ImportError:  # python rl/sample_pool.py
    from agent_env import HOLDOUT_SEED0, generate_task, run_episode
    from agent_rollout import OllamaPolicy, control_policy, oracle_policy
    from build_pairs import HEADER, body
    from verifier import grade

ADV_EPS = 1e-4


# --------------------------------------------------------------------------- group
def group_advantage(rewards):
    """(mean, std, [A_i]) — the group-normalised advantage (RL_PLAN_2 §3).

    Population std, because the group is the whole population. `+eps` keeps a
    numerically-nonzero group finite; a truly zero-std group is flagged degenerate
    and its advantage ignored, so the eps never fabricates a gradient.
    """
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    advantages = [(r - mean) / (std + ADV_EPS) for r in rewards]
    return mean, std, advantages


def rows_from_episode(ep, task_group_id, reward, advantage, meta):
    """One episode -> its turn-level GRPO rows.

    `group_id` is the **task's** group (the G episodes of one task); `episode_id`
    identifies the single rollout within it. The advantage is per-episode, the
    group is what `train_grpo.rank_rows` keeps whole and `grpo_report` measures.

    `prompt_msgs` + `completion` is exactly the Stage-A row shape (`build_pairs`),
    so `sft/rollout.render_prefix` and the byte invariant carry over unchanged.
    """
    turns = []                     # (index_in_messages, stopped)
    stopped_flags = ep.get("turn_stopped") or []
    k = 0
    for i, m in enumerate(ep["messages"]):
        if m.get("role") == "assistant":
            stopped = bool(stopped_flags[k]) if k < len(stopped_flags) else True
            turns.append((i, stopped))
            k += 1
    episode_id = f"{task_group_id}:g{meta['group_index']}"
    rows = []
    for turn_index, (i, stopped) in enumerate(turns):
        rows.append({
            "key": f"{episode_id}:t{turn_index}",
            "group_id": task_group_id,
            "episode_id": episode_id,
            "task_id": ep["task_id"],
            "group_index": meta["group_index"],
            "turn_index": turn_index,
            "prompt_msgs": ep["messages"][:i],
            "completion": HEADER + body(ep["messages"][i]),
            "completion_stopped": stopped,
            "reward": reward,
            "advantage": advantage,
            "submitted": bool(ep.get("submitted")),
            "group_mean": meta["group_mean"],
            "group_std": meta["group_std"],
            "group_size": meta["group_size"],
            "degenerate": meta["degenerate"],
            "tier": meta["tier"],
            "anti_hack": meta["anti_hack"],
            "logpi_ref": None,      # filled by rl/refpass.py on the torch box
        })
    return rows


def sample_task(task, policy_of_group, group, keep=False):
    """Run G episodes, grade, and return (rows, group_summary)."""
    eps = []
    for g in range(group):
        ep = run_episode(task, policy_of_group(task, g), keep=keep, grader=grade)
        eps.append(ep)
    rewards = [ep["grade"]["reward"] for ep in eps]
    mean, std, advantages = group_advantage(rewards)
    degenerate = std == 0.0
    rows, summary = [], {
        "task_id": task.task_id, "rewards": rewards, "mean": round(mean, 4),
        "std": round(std, 4), "degenerate": degenerate,
        "tiers": [ep["grade"]["tier"] for ep in eps],
        "anti_hack": sorted({a for ep in eps for a in ep["grade"]["anti_hack"]}),
        "pass_rate": round(sum(1 for r in rewards if r == 1.0) / group, 3),
        "submit_rate": round(sum(1 for ep in eps if ep["submitted"]) / group, 3),
    }
    for g, ep in enumerate(eps):
        meta = {"group_index": g, "group_mean": mean, "group_std": std,
                "group_size": group, "degenerate": degenerate,
                "tier": ep["grade"]["tier"], "anti_hack": ep["grade"]["anti_hack"]}
        rows.extend(rows_from_episode(ep, task.task_id, rewards[g], advantages[g],
                                      meta))
    return rows, summary


# --------------------------------------------------------------------------- policies
def make_policy(kind, args):
    """`policy_of_group(task, g) -> policy`. Kept a factory so a scripted policy
    can vary by group (the selftest's oracle/control mix)."""
    if kind == "ollama":
        pol = OllamaPolicy(args.model, args.url, temp=args.temp,
                           num_predict=args.num_predict, num_ctx=args.num_ctx)
        return lambda task, g: pol
    if kind == "oracle":
        return lambda task, g: oracle_policy(task)
    if kind == "control":
        return lambda task, g: control_policy
    if kind == "mixed":                      # selftest only: forces a reward spread
        return lambda task, g: (oracle_policy(task) if g % 2 == 0 else control_policy)
    raise SystemExit(f"unknown policy {kind}")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, default=100, help="starting tasks to sample")
    ap.add_argument("--group", type=int, default=16, help="G episodes per task")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "tinyballs-dpo"))
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL",
                                                    "http://127.0.0.1:11434/api/chat"))
    ap.add_argument("--policy", choices=["ollama", "oracle", "control", "mixed"],
                    default="ollama")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--num-predict", type=int, default=512)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--target-groups", type=int, default=0,
                    help="dynamically refill until this many NON-degenerate groups")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--out", default="rl/samples/batch-0001.jsonl")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted pool: skip seeds already present "
                         "in --out (implies --append)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    policy_of_group = make_policy(a.policy, a)
    seed0 = a.seed0 + (HOLDOUT_SEED0 if a.holdout else 0)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # Resume: tasks are deterministic in `seed0 + i`, so the count of distinct
    # groups already written tells us where to continue. Keeps a crashed 30-min
    # Mac generation from throwing away the groups it already finished.
    start = 0
    if a.resume and os.path.exists(a.out):
        seen = set()
        for line in open(a.out):
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["group_id"])
            except Exception:  # noqa: BLE001
                pass
        start = len(seen)
        print(f"  resume: {start} groups already in {a.out}; "
              f"continuing at seed {seed0 + start}", flush=True)

    n_rows = n_groups = n_degen = 0
    max_tasks = a.max_tasks or (a.tasks if not a.target_groups else a.tasks * 20)
    t0 = time.time()
    fh = open(a.out, "a" if (a.append or a.resume) else "w")
    try:
        for i in range(start, max_tasks):
            task = generate_task(seed0 + i)
            rows, summary = sample_task(task, policy_of_group, a.group, keep=a.keep)
            n_groups += 1
            if summary["degenerate"]:
                n_degen += 1
            good = n_groups - n_degen
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += len(rows)
            fh.flush()
            print(f"  {task.task_id:<16} rewards={summary['rewards']} "
                  f"std={summary['std']:.3f} pass={summary['pass_rate']} "
                  f"sub={summary['submit_rate']} "
                  f"tiers={sorted(set(summary['tiers']))} "
                  f"{'DROP' if summary['degenerate'] else ''}", flush=True)
            if a.target_groups and good >= a.target_groups:
                break
            if not a.target_groups and (i + 1) >= a.tasks:
                break
    finally:
        fh.close()

    dt = time.time() - t0
    print(f"\n{n_groups} groups ({n_degen} degenerate, "
          f"{n_degen/max(1,n_groups):.1%} dropped) -> {n_rows} rows in {a.out} "
          f"({dt/60:.1f} min, {dt/max(1,n_groups):.1f}s/group)")
    print("GATE G0: run `python rl/grpo_report.py --samples " + a.out + "` before "
          "spending GPU time.")
    return 0 if n_rows else 2


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    """Gate G0 without a model: the mixed policy must produce live groups and
    the group maths must be exactly zero-mean."""
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== group_advantage: zero-mean, and a flat group is degenerate")
    m, s, adv = group_advantage([1.0, 0.0, 0.0, 1.0])
    ok("adv_zero_mean", abs(sum(adv)) < 1e-6, str(adv))
    ok("adv_spread", s > 0)
    m2, s2, adv2 = group_advantage([0.6, 0.6, 0.6])
    ok("flat_std_zero", s2 == 0.0)
    ok("flat_adv_zero", all(abs(x) < 1e-2 for x in adv2), str(adv2))

    print("== a mixed group yields rows with the trainer's shape")
    tasks = [generate_task(s) for s in range(3)]
    rows, summary = sample_task(tasks[0], make_policy("mixed", None), 4)
    ok("mixed_not_degenerate", summary["degenerate"] is False, str(summary))
    ok("reward_spread", len(set(summary["rewards"])) > 1, str(summary["rewards"]))
    p = rows[0]
    ok("row_keys", set(p) >= {"group_id", "episode_id", "task_id", "turn_index",
                              "prompt_msgs", "completion", "reward", "advantage",
                              "group_mean", "group_std", "degenerate", "logpi_ref"})
    ok("completion_header", p["completion"].startswith(HEADER))
    ok("prompt_is_prefix", all(m["role"] != "assistant" for m in p["prompt_msgs"]))
    advs = {r["episode_id"]: r["advantage"] for r in rows}
    ok("one_advantage_per_episode", len(set(advs.values())) >= 2)
    sum_by_ep = {}
    for r in rows:
        sum_by_ep[r["episode_id"]] = r["advantage"]
    ok("adv_zero_mean_over_eps", abs(sum(sum_by_ep.values())) < 1e-6,
       str(sum_by_ep))

    print("== an all-oracle pool is degenerate (the trainer would drop it)")
    rows2, summary2 = sample_task(tasks[1], make_policy("oracle", None), 3)
    ok("oracle_all_pass", summary2["rewards"] == [1.0, 1.0, 1.0])
    ok("oracle_degenerate", summary2["degenerate"] is True)
    ok("oracle_rows_flagged",
       all(r["degenerate"] for r in rows2), "rows must carry the flag too")

    print("== anti-hack rides along in the summary (for the report)")
    rows3, summary3 = sample_task(tasks[2], make_policy("control", None), 3)
    ok("control_zero", summary3["rewards"] == [0.0, 0.0, 0.0])
    ok("control_tier_none", set(summary3["tiers"]) == {"none"}, str(summary3["tiers"]))

    print()
    if fails:
        print(f"SAMPLE_POOL SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("SAMPLE_POOL SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
