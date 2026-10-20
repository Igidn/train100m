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
import re
import socket
import time
import urllib.error
import urllib.request

try:
    from .agent_env import (TOOLS, HOLDOUT_SEED0, calls_with_results, generate_task,
                            mutation_applied, parse_calls, run_episode, system_prompt)
except ImportError:  # run as a plain script (python rl/agent_rollout.py)
    from agent_env import (TOOLS, HOLDOUT_SEED0, calls_with_results, generate_task,
                           mutation_applied, parse_calls, run_episode, system_prompt)
try:
    from .agent_env import IM_END
except ImportError:
    from agent_env import IM_END

# What training put in the *completion*: the SFT target starts at the header, so
# a raw decode from a prompt ending on `<|im_end|>` returns it as part of the
# generation (`sft.rollout.make_responder` does not add a generation prompt).
ASSISTANT_HEADER = "<|im_start|>assistant\n"

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
# A long Mac generation (100 tasks x G=16) holds the Ollama server for ~30 min;
# a single transient 5xx/connection reset must not kill the whole pool, so the
# serving client retries with backoff and, if it still fails, fails soft (one
# empty turn) rather than raising out of the episode loop.
TRANSIENT_RETRIES = int(os.environ.get("OLLAMA_RETRIES", "5"))
DECLARED = {"bash": ("command",),
            "str_replace_editor": ("command", "path"),
            "submit": ()}
# per-command required args: `command`/`path` alone let a `str_replace` with no
# `new_str` count as valid (round 5). See `agent_env.execute_tool`.
REQUIRED = {"view": (), "create": ("file_text",),
            "str_replace": ("old_str", "new_str"), "insert": ("new_str",)}
EDITOR_MUTATE = ("create", "str_replace", "insert")


# --------------------------------------------------------------------------- policies
class OllamaPolicy:
    """Student on Ollama. Native tool_calls when parsed, else content JSON; the
    env's run_episode reconstructs the trained `<|tool_call|>` history for us."""

    def __init__(self, model, url=OLLAMA, temp=0.7, top_p=0.95,
                 num_predict=512, num_ctx=16384, timeout=600, send_tools=False,
                 seed=None):
        self.model, self.url, self.temp, self.top_p = model, url, temp, top_p
        self.num_predict, self.num_ctx, self.timeout = num_predict, num_ctx, timeout
        self.seed = seed
        # The Modelfile template appends its own "Available tools" block when
        # `$.Tools` is set, and our system prompt already embeds the tool JSON —
        # sending both duplicates them (RL_PLAN §4.3, serving consistency).
        self.send_tools = send_tools

    def __call__(self, messages):
        opts = {"temperature": self.temp, "top_p": self.top_p,
                "num_predict": self.num_predict, "num_ctx": self.num_ctx}
        # Ollama samples from a global RNG unless a seed is given, so without this
        # two eval "seeds" were just disjoint task ranges, not replications.
        if self.seed is not None:
            opts["seed"] = int(self.seed)
        payload = {"model": self.model, "messages": messages, "stream": False,
                   "options": opts}
        if self.send_tools:
            payload["tools"] = TOOLS
        body = json.dumps(payload).encode()
        last_err = None
        for attempt in range(TRANSIENT_RETRIES):
            try:
                req = urllib.request.Request(
                    self.url, data=body,
                    headers={"Content-Type": "application/json"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.load(r)
                dt = time.time() - t0
                break
            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:      # our bug (bad model/args): do not loop
                    raise
                last_err = e
            except (urllib.error.URLError, socket.timeout, OSError) as e:
                last_err = e
            wait = min(2 ** attempt, 30)
            print(f"  ! ollama transient error ({last_err}); retry "
                  f"{attempt + 1}/{TRANSIENT_RETRIES - 1} in {wait}s", flush=True)
            time.sleep(wait)
        else:
            # Fail soft: end this one episode and let the verifier grade it (it
            # lands on the 0.0 rung). Killing a 1,600-episode pool over one turn
            # is the worse outcome.
            print(f"  ! ollama gave up after {TRANSIENT_RETRIES} tries "
                  f"({last_err}); ending episode early", flush=True)
            return {"text": "", "stopped": True, "calls": [], "secs": 0.0,
                    "prompt_tok": 0, "eval_tok": 0, "error": str(last_err)}
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


class HFPolicy:
    """The same student, served in-process from a checkpoint (no daemon).

    Ollama is the Mac's serving path. A Kaggle kernel has no daemon, and
    pip-installing vLLM into the kernel's interpreter breaks the CUDA runtime
    (`sft/vllm_phase.py` runs it as a child for that reason), so a kernel drives
    the weights through the KV engine the SFT rollout already uses. Both paths
    go through `sft/rollout.render_prefix`, so the ids the model sees are the
    same on the Mac and on Kaggle — the deployment lesson from round 5.

    `model_dir` must be HF format (`llama/export_hf.py` writes it); a bare
    trainer `.pt` is a state dict, not a directory a tokenizer can be read from.
    """

    def __init__(self, model_dir=None, repo=None, device=None, max_len=16384,
                 temp=0.7, top_p=0.95, max_tokens=512, seed=1234):
        from sft.rollout import KVEngine, make_responder, render_prefix
        self.engine = KVEngine(model_dir=model_dir, repo=repo, device=device,
                              max_len=max_len)
        self.respond = make_responder(self.engine, self.engine.tok,
                                      max_tokens=max_tokens, temp=temp,
                                      top_p=top_p, seed=seed)
        self.render_prefix = render_prefix
        self.max_len = max_len

    def __call__(self, messages):
        t0 = time.time()
        # The engine truncates the prefix from the left (`p[-max_len:]`) without
        # saying so, which on a real repo means the system prompt and the issue
        # silently disappear. Measure the real length and report it, so the pair
        # generator can refuse an instance instead of training on the tail of a
        # source file. Encoding twice costs ~ms against a forward pass.
        n_prompt = len(self.render_prefix(self.engine.tok, messages))
        text, stopped = self.respond(messages)
        text = text or ""
        # The header is part of the *completion* in training, so the engine
        # hands it back; the env's history and `build_pairs.completion` both add
        # it themselves, so leaving it in would double it.
        if text.startswith(ASSISTANT_HEADER):
            text = text[len(ASSISTANT_HEADER):]
        if text.endswith(IM_END):
            text = text[:-len(IM_END)]
        return {"text": text.strip("\n"), "stopped": bool(stopped),
                "calls": [], "secs": time.time() - t0,
                "prompt_tok": n_prompt,
                "eval_tok": 0,
                "truncated": n_prompt > self.max_len}


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


def _calls_with_results(ep):
    """[(name, args, result)] — the pairing `agent_env.calls_with_results` builds.

    The result is needed so `verify_with_check` counts a check run after an edit
    that was actually *applied*: round 5 accepted any `str_replace` call, so 30 of
    the 66 "verified" R5 episodes had no successful edit at all.
    """
    return calls_with_results(ep.get("messages") or [])


def _arg_ok(name, args):
    """The call carries every argument its own command requires.

    Round 5's check only named `command`/`path`, so `tool_valid=0.973` never
    noticed a `str_replace` with no `new_str`. Required sets are per command.
    """
    keys = DECLARED.get(name)
    if keys is None or not isinstance(args, dict):
        return False
    for k in keys:
        v = args.get(k)
        if not isinstance(v, str) or not v.strip():
            return False
    if name == "str_replace_editor":
        cmd = args.get("command")
        if cmd not in ("view", "create", "str_replace", "insert"):
            return False
        need = REQUIRED.get(cmd, ())
        for k in need:
            v = args.get(k)
            if not isinstance(v, str) or not v.strip():
                return False
        if cmd == "insert" and not (isinstance(args.get("insert_line"), int) or
                                    str(args.get("insert_line", "")).strip().lstrip("-").isdigit()):
            return False
    return True


def behavior_metrics(episodes):
    """The RL_PLAN §0 targets, computed from recorded trajectories.

    `verify_with_check` is the §0.1 correction of the weak `verify_before_submit`:
    it asks whether the episode ran *the task's own check command* after its last
    edit, not whether it ran "any bash". Of the 57 Stage-A submits, 20 "verified"
    with a bare `ls`, which the old metric counted as a pass.
    """
    n = len(episodes) or 1
    valid_n = valid_d = 0
    rbe_hit = rbe_den = 0
    vbs_hit = vbs_den = 0
    vbc_hit = vbc_den = 0
    broken = 0
    for ep in episodes:
        seq = _calls_with_results(ep)
        if re.search(r"(SyntaxError|IndentationError|TabError)",
                     (ep.get("check") or {}).get("output") or ""):
            broken += 1
        viewed, last_edit, last_bash = set(), -1, -1
        for i, (name, args, res) in enumerate(seq):
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
                    # only an *applied* edit counts as "the last edit" — round 5
                    # let a failed call stand in, which made verify_with_check
                    # count a check over an unchanged file
                    if mutation_applied(args, res):
                        last_edit = i
            if name == "bash":
                last_bash = i
        if last_edit >= 0:
            vbs_den += 1
            vbs_hit += 1 if last_bash > last_edit else 0
            check_cmd = str((ep.get("check") or {}).get("cmd", "")).strip()
            vbc_den += 1
            if check_cmd and any(
                    name == "bash" and str(a.get("command", "")).strip() == check_cmd
                    for name, a, _r in seq[last_edit + 1:]):
                vbc_hit += 1

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
        "verify_with_check": rate(vbc_hit, vbc_den),
        "syntax_broken_rate": rate(broken, n),
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
    ap.add_argument("--holdout", action="store_true",
                    help="use the held-out eval seed range (RL_PLAN_2 §6.1)")
    ap.add_argument("--out", default="trajectories.jsonl")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--gen-seed", type=int, default=None,
                    help="sampling seed for Ollama (makes eval runs repeatable)")
    ap.add_argument("--send-tools", action="store_true",
                    help="also pass tools= to Ollama (do NOT combine with the "
                         "system-embedded tools unless the Modelfile differs)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    tasks = [generate_task(a.seed0 + i + (HOLDOUT_SEED0 if a.holdout else 0))
             for i in range(a.tasks)]

    if a.dry:
        print(system_prompt("<workdir>"))
        print("\n---- user ----\n" + tasks[0].user_prompt("<workdir>"))
        return 0

    if a.control:
        label, policy_fn = "control", (lambda t: control_policy)
    elif a.oracle:
        label, policy_fn = "oracle", oracle_policy
    else:
        pol = OllamaPolicy(a.model, a.url, temp=a.temp, send_tools=a.send_tools,
                           seed=a.gen_seed)
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
    ok("gold.verify_with_check=1", mg["verify_with_check"] == 1.0)
    ok("gold.loop=0", mg["loop_rate"] == 0.0)
    ok("gold.tool_valid=1", mg["tool_valid"] == 1.0)
    ok("ctrl.check_pass=0", mc["check_pass_rate"] == 0.0)
    ok("ctrl.terminate=1", mc["terminate_rate"] == 1.0)

    print("== no-op policy (no calls) does not pass")
    noop = run_episode(tasks[0], lambda m: {"text": "nothing", "stopped": True, "calls": []})
    ok("noop.check_false", noop["check"]["passed"] is False)
    ok("noop.steps=1", noop["steps"] == 1)

    print("== round 6: a str_replace with no new_str is an invalid tool call")
    path = tasks[0].fix["path"]
    tc = "<" + "|tool_call|>"
    bad = [run_episode(tasks[0], _steps_policy([
        {"text": tc + json.dumps({"name": "str_replace_editor",
                                  "arguments": {"command": "view", "path": path}}),
         "stopped": True, "calls": []},
        {"text": tc + json.dumps({"name": "str_replace_editor",
                                  "arguments": {"command": "str_replace",
                                                "path": path,
                                                "old_str": tasks[0].fix["old_str"]}}),
         "stopped": True, "calls": []},
        {"text": tc + json.dumps({"name": "bash",
                                  "arguments": {"command": tasks[0].check_cmd}}),
         "stopped": True, "calls": []},
    ]))]
    mb = behavior_metrics(bad)
    print("  bad:", mb)
    ok("bad.tool_valid_lt_1", mb["tool_valid"] < 1.0, str(mb["tool_valid"]))
    ok("bad.not_verified", mb["verify_with_check"] is None or mb["verify_with_check"] == 0.0,
       str(mb["verify_with_check"]))
    ok("bad.no_pass", mb["check_pass_rate"] == 0.0)

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
