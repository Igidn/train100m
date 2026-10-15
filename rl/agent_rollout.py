#!/usr/bin/env python3
"""Rollout harness for the agentic RL stage — rl/RL_PLAN.md §5.

Drives `rl/agent_env.py` with one of three policies and records trajectories:

  --model M      Ollama /api/chat (the student, GGUF, torch-free)
  --oracle       a deterministic scripted solver built from each task's known
                 fix — produces the *gold* trajectory for Stage A `chosen`, and
                 doubles as the no-model smoke test
  --control      the do-nothing control (RL_PLAN §3.4): submit immediately,
                 assert it does NOT pass

Plus per-episode trajectory records and the behavioral metrics the plan commits
to (tool-valid, read-before-edit, verify-before-submit, loop, terminate, check).

Self-test (no model, no Ollama):
    python rl/agent_rollout.py --selftest
Smoke the whole env without a model:
    python rl/agent_rollout.py --oracle --tasks 8 --out gold.jsonl
Live, once Ollama is up:
    python rl/agent_rollout.py --model tinyballs --tasks 16 --out rollouts.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

try:
    from .agent_env import TOOLS, generate_task, parse_calls, run_episode, system_prompt
except ImportError:  # run as a plain script (python rl/agent_rollout.py)
    from agent_env import TOOLS, generate_task, parse_calls, run_episode, system_prompt

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
DECLARED = {"bash": ("command",),
            "str_replace_editor": ("command", "path"),
            "submit": ()}
EDITOR_MUTATE = ("create", "str_replace", "insert")


# --------------------------------------------------------------------------- policies
class OllamaPolicy:
    """Student on Ollama. Native tool_calls when parsed, else content JSON; the
    env's run_episode reconstructs the trained `<|tool_call|>` history for us."""

    def __init__(self, model, url=OLLAMA, temp=0.7, top_p=0.95,
                 num_predict=512, num_ctx=16384, timeout=600, send_tools=False):
        self.model, self.url, self.temp, self.top_p = model, url, temp, top_p
        self.num_predict, self.num_ctx, self.timeout = num_predict, num_ctx, timeout
        # The Modelfile template appends its own "Available tools" block when
        # `$.Tools` is set, and our system prompt already embeds the tool JSON —
        # sending both duplicates them (RL_PLAN §4.3, serving consistency).
        self.send_tools = send_tools

    def __call__(self, messages):
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": {"temperature": self.temp, "top_p": self.top_p,
                               "num_predict": self.num_predict, "num_ctx": self.num_ctx}}
        if self.send_tools:
            payload["tools"] = TOOLS
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.load(r)
        dt = time.time() - t0
        msg = resp.get("message") or {}
        content = msg.get("content") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append((fn.get("name"), args))
        reason = resp.get("done_reason") or "stop"
        return {"text": content, "stopped": reason != "length", "calls": calls,
                "secs": dt, "prompt_tok": resp.get("prompt_eval_count", 0),
                "eval_tok": resp.get("eval_count", 0)}


def _steps_policy(steps):
    box = {"i": 0}

    def policy(messages):
        i = box["i"]
        box["i"] += 1
        return steps[i] if i < len(steps) else {"text": "", "stopped": True, "calls": []}
    return policy


def oracle_policy(task):
    """The gold trajectory for a micro-task: inspect, edit, verify, submit.
    This is the `chosen` side of Stage A with no model and no environment guesswork."""
    tc = "<" + "|tool_call|>"

    def call(name, **args):
        return tc + json.dumps({"name": name, "arguments": args})
    fix = task.fix or {}
    path = fix.get("path", "")
    return _steps_policy([
        {"text": call("bash", command="ls"), "stopped": True, "calls": []},
        {"text": call("str_replace_editor", command="view", path=path),
         "stopped": True, "calls": []},
        {"text": call("str_replace_editor", command="str_replace", path=path,
                      old_str=fix.get("old_str", ""), new_str=fix.get("new_str", "")),
         "stopped": True, "calls": []},
        {"text": call("bash", command=task.check_cmd), "stopped": True, "calls": []},
        {"text": call("submit"), "stopped": True, "calls": []},
    ])


def control_policy(messages):
    tc = "<" + "|tool_call|>"
    return {"text": "I'm not sure how to fix this. " + tc + '{"name": "submit", "arguments": {}}',
            "stopped": True, "calls": None}


# --------------------------------------------------------------------------- metrics
def _calls_in_order(ep):
    seq = []
    for m in ep.get("messages", []):
        if m.get("role") == "assistant":
            seq.extend(parse_calls(m.get("content") or ""))
    return seq


def _arg_ok(name, args):
    keys = DECLARED.get(name)
    if keys is None or not isinstance(args, dict):
        return False
    for k in keys:
        v = args.get(k)
        if not isinstance(v, str) or not v.strip():
            return False
    if name == "str_replace_editor" and args.get("command") not in \
            ("view", "create", "str_replace", "insert"):
        return False
    return True


def behavior_metrics(episodes):
    """The RL_PLAN §0 targets, computed from recorded trajectories."""
    n = len(episodes) or 1
    valid_n = valid_d = 0
    rbe_hit = rbe_den = 0
    vbs_hit = vbs_den = 0
    for ep in episodes:
        seq = _calls_in_order(ep)
        viewed, last_edit, last_bash = set(), -1, -1
        for i, (name, args) in enumerate(seq):
            if name in DECLARED:
                valid_d += 1
                valid_n += 1 if _arg_ok(name, args) else 0
            if name == "str_replace_editor":
                cmd, p = args.get("command"), str(args.get("path", ""))
                if cmd == "view":
                    viewed.add(p)
                elif cmd in EDITOR_MUTATE:
                    rbe_den += 1
                    rbe_hit += 1 if p in viewed else 0
                    last_edit = i
            if name == "bash":
                last_bash = i
        if last_edit >= 0:
            vbs_den += 1
            vbs_hit += 1 if last_bash > last_edit else 0

    def rate(a, b):
        return round(a / b, 3) if b else None
    return {
        "n": len(episodes),
        "check_pass_rate": rate(sum(1 for e in episodes if e["check"]["passed"]), n),
        "terminate_rate": rate(sum(1 for e in episodes if e["submitted"]), n),
        "loop_rate": rate(sum(1 for e in episodes if e["loop"]), n),
        "tool_valid": rate(valid_n, valid_d),
        "read_before_edit": rate(rbe_hit, rbe_den),
        "verify_before_submit": rate(vbs_hit, vbs_den),
        "mean_steps": round(sum(e["steps"] for e in episodes) / n, 2),
    }


# --------------------------------------------------------------------------- drive
def run(tasks, policy_fn, out_path, keep=False, label=""):
    episodes = []
    with open(out_path, "w") as fh:
        for task in tasks:
            ep = run_episode(task, policy_fn(task), keep=keep)
            ep["label"] = label
            fh.write(json.dumps(ep, default=str) + "\n")
            episodes.append(ep)
            print(f"  {task.task_id:<16} steps={ep['steps']} "
                  f"check={'PASS' if ep['check']['passed'] else 'fail'} "
                  f"loop={ep['loop']} submitted={ep['submitted']}")
    return episodes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "tinyballs"))
    ap.add_argument("--url", default=OLLAMA)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--out", default="trajectories.jsonl")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--send-tools", action="store_true",
                    help="also pass tools= to Ollama (do NOT combine with the "
                         "system-embedded tools unless the Modelfile differs)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    tasks = [generate_task(a.seed0 + i) for i in range(a.tasks)]

    if a.dry:
        print(system_prompt("<workdir>"))
        print("\n---- user ----\n" + tasks[0].user_prompt("<workdir>"))
        return 0

    if a.control:
        label, policy_fn = "control", (lambda t: control_policy)
    elif a.oracle:
        label, policy_fn = "oracle", oracle_policy
    else:
        pol = OllamaPolicy(a.model, a.url, temp=a.temp, send_tools=a.send_tools)
        label, policy_fn = a.model, (lambda t: pol)

    print(f"rolling out {len(tasks)} tasks with {label} -> {a.out}")
    eps = run(tasks, policy_fn, a.out, keep=a.keep, label=label)
    print("\n" + json.dumps(behavior_metrics(eps), indent=1))
    return 0


def selftest() -> int:
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== oracle policy solves generated tasks; control does not")
    tasks = [generate_task(s) for s in range(6)]
    gold = [run_episode(t, oracle_policy(t)) for t in tasks]
    ctrl = [run_episode(t, control_policy) for t in tasks]
    mg, mc = behavior_metrics(gold), behavior_metrics(ctrl)
    print("  gold:", mg)
    print("  ctrl:", mc)
    ok("gold.check_pass=1", mg["check_pass_rate"] == 1.0)
    ok("gold.read_before_edit=1", mg["read_before_edit"] == 1.0)
    ok("gold.verify_before_submit=1", mg["verify_before_submit"] == 1.0)
    ok("gold.loop=0", mg["loop_rate"] == 0.0)
    ok("gold.tool_valid=1", mg["tool_valid"] == 1.0)
    ok("ctrl.check_pass=0", mc["check_pass_rate"] == 0.0)
    ok("ctrl.terminate=1", mc["terminate_rate"] == 1.0)

    print("== no-op policy (no calls) does not pass")
    noop = run_episode(tasks[0], lambda m: {"text": "nothing", "stopped": True, "calls": []})
    ok("noop.check_false", noop["check"]["passed"] is False)
    ok("noop.steps=1", noop["steps"] == 1)

    print()
    if fails:
        print(f"SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
