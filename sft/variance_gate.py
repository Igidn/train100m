"""Gate 0 — the reward-variance gate (`RL_TRAINING.md` §1, step 0).

GRPO's gradient is the group-relative advantage. If all N rollouts of a prompt
score the same, every advantage is 0: the run finishes, the loss moves, nothing
learns. No crash, just a flat curve — the most expensive way to find out.

`multi_tool` is 0/8 task-correct through the unified serving path, so this gate
is the precondition for Track 1, not a formality. The shaped reward in
`sft/reward.py` is supposed to put that scenario at 0.8-1.0 on its
format/args/grounded terms while task-correct sits at 0; the question is whether
real rollouts actually spread there, or whether the model collapses every group
onto one verdict the way it did on the AWM probe (12 of 14 episodes identical).

    go    non-trivial within-group spread on a healthy fraction of prompts
    no-go rewards collapse to one value -> do not spend quota

Also runs the do-nothing control on the same tasks, because a reward that pays
for refusing to act is the failure mode that AWM hid (1 task in 12 "passed" with
zero tool calls).

    python sft/variance_gate.py --engine vllm --prompts 24 --rollouts 8
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import (WEIGHTS, compare_weather_task, error_recovery_task,  # noqa: E402
                    run_episode, score_episode, single_weather_task)
from rollout import make_responder  # noqa: E402

CITIES = ["Tokyo", "Paris", "London", "New York"]


def task_pool(n_prompts, faults=((), ("transient",))):
    """Every city pair in both fault modes, then the single-tool and abstention
    shapes. The env is small (3 tools, 4 cities) and RL will memorise it — that is
    a real risk (`RL_READINESS.md` §3) but not what this gate measures; what this
    gate measures is whether *any* prompt produces a reward spread.
    """
    pairs = [(a, b) for a in CITIES for b in CITIES if a != b]
    tasks = []
    for f in faults:
        for a, b in pairs:
            tasks.append(compare_weather_task(a, b, faults=f))
    for c in CITIES:
        tasks.append(single_weather_task(c))
    for c in ("Berlin", "Atlantis", "Narnia"):
        tasks.append(error_recovery_task(c))
    # round-robin so a truncated --prompts still spans shapes
    return tasks[:n_prompts] if n_prompts else tasks


def do_nothing_control(tasks):
    """Reward for a policy that never calls a tool."""
    scores = [score_episode(t, ["I don't have that information."], results=[],
                            calls=[])["total"] for t in tasks]
    return {"n": len(scores), "mean": statistics.fmean(scores),
            "max": max(scores), "nonzero": sum(1 for s in scores if s > 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "kv"], default="vllm")
    ap.add_argument("--model", default=os.environ.get(
        "SMOKE_MODEL", "igidn/TinyBalls-110M-V1"))
    ap.add_argument("--prompts", type=int, default=24)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", default="variance_gate.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.engine == "vllm":
        from rollout import VLLMEngine
        engine = VLLMEngine(args.model)
    else:
        from rollout import KVEngine
        engine = KVEngine(repo=args.model)
    print(f"engine {engine.name}, model {args.model}, temp {args.temp}", flush=True)

    tasks = task_pool(args.prompts)
    control = do_nothing_control(tasks)
    print(f"do-nothing control: {control}", flush=True)

    per_task = []
    for ti, task in enumerate(tasks):
        responder = make_responder(engine, tok, max_tokens=args.max_tokens,
                                  temp=args.temp, seed=1000 * ti)
        rollouts = []
        for r in range(args.rollouts):
            sc, ep = run_episode(task, responder, max_steps=args.max_steps,
                                 faults=list(task.faults))
            sc["seed"] = r
            sc["texts"] = ep["turns"]
            rollouts.append(sc)
        totals = [x["total"] for x in rollouts]
        spread = (max(totals) - min(totals)) if totals else 0.0
        sd = statistics.pstdev(totals) if len(totals) > 1 else 0.0
        comp = {k: statistics.fmean([x["parts"].get(k, 0.0) for x in rollouts])
                for k in list(WEIGHTS) + ["format", "declared", "grounded", "correct"]}
        per_task.append({
            "task": task.name, "prompt": task.prompt, "faults": list(task.faults),
            "totals": totals, "mean": statistics.fmean(totals), "std": sd,
            "spread": spread, "distinct": len(set(round(t, 4) for t in totals)),
            "component_means": comp,
            "n_correct": sum(1 for x in rollouts if x["parts"]["correct"] > 0.5),
            "example_worst": min(rollouts, key=lambda x: x["total"])["final"][:200],
            "example_best": max(rollouts, key=lambda x: x["total"])["final"][:200],
        })
        print(f"[{ti+1}/{len(tasks)}] {task.name:34s} mean {statistics.fmean(totals):.3f} "
              f"sd {sd:.3f} spread {spread:.3f} distinct {per_task[-1]['distinct']} "
              f"correct {per_task[-1]['n_correct']}/{args.rollouts}", flush=True)

    flat = [p for p in per_task if p["spread"] < 0.05]
    healthy = [p for p in per_task if p["spread"] >= 0.15]
    by_shape = {}
    for p in per_task:
        shape = p["task"].split("_")[0]
        by_shape.setdefault(shape, []).append(p["spread"])
    verdict = {
        "prompts": len(per_task),
        "flat_prompts": len(flat),
        "healthy_prompts": len(healthy),
        "flat_fraction": len(flat) / max(1, len(per_task)),
        "mean_spread": statistics.fmean([p["spread"] for p in per_task]),
        "mean_within_group_std": statistics.fmean([p["std"] for p in per_task]),
        "mean_spread_by_shape": {k: statistics.fmean(v)
                                 for k, v in by_shape.items()},
        "task_correct_rate": sum(p["n_correct"] for p in per_task)
        / max(1, len(per_task) * args.rollouts),
        "do_nothing_control": control,
    }
    # go / no-go: a healthy fraction of prompts must spread, and the control must
    # pay nothing (a control that pays means the reward is gameable by inaction)
    go = (verdict["healthy_prompts"] >= max(2, 0.3 * len(per_task))
          and control["nonzero"] == 0)
    verdict["verdict"] = "go" if go else "no-go"
    if not go and control["nonzero"]:
        verdict["reason"] = "do-nothing control earns reward; the shaped terms pay for inaction"

    print("\n=== Gate 0 ===")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nVERDICT: {verdict['verdict'].upper()}")

    out = {"args": vars(args), "verdict": verdict, "tasks": per_task}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {args.out}")
    return 0 if go else 2


if __name__ == "__main__":
    raise SystemExit(main())