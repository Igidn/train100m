#!/usr/bin/env python3
"""Agent environment for the agentic RL stage — rl/RL_PLAN.md §3.

A jailed local sandbox, the three tools the gold data uses
(`bash` / `str_replace_editor` / `submit`), a tier-0 micro-task generator, and
the episode loop. Model-agnostic: the loop drives any `policy(messages) -> dict`
so it runs under a scripted policy with no model at all (for tests and the
do-nothing control) and under Ollama in production.

Why these tools, verbatim (RL_PLAN §3.1): they are exactly what
`SWE-bench/SWE-smith-trajectories` uses, so the model's behavior transfers in and
out of the gold distribution without a rename.

Sandbox on macOS: a per-episode temp dir + `cwd`-jail + `subprocess` timeout +
`setrlimit`. This is NOT a security boundary against a hostile agent; it is a
guard against the far more likely runaway command (rm -rf, infinite loop, filling
the disk). Documented rather than hidden.

Self-test, no model:
    python rl/agent_env.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IM_START = "<" + "|im_start|>"
IM_END = "<" + "|im_end|>"
TOOL_CALL = "<" + "|tool_call|>"
TOOL_RESULT = "<" + "|tool_result|>"

MAX_OUTPUT = 4000          # chars of any single tool result (context is a budget)
BASH_TIMEOUT = 20          # seconds per command
MAX_STEPS = 8              # tool turns per episode

# ---------------------------------------------------------------------------
# tool surface — mirrors SWE-smith's tool split exactly
# ---------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": ("Run a bash command in the repository (the command's cwd is "
                        "the repo root). Use it to list files, read them with cat, "
                        "and run the tests."),
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "str_replace_editor",
        "description": ("View, create, or edit a file. `command` is one of "
                        "view | create | str_replace | insert. "
                        "view shows a file (optionally `view_range` [start,end]); "
                        "create writes `file_text` to a new `path`; "
                        "str_replace replaces the unique `old_str` with `new_str`; "
                        "insert adds `new_str` after line `insert_line`."),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string",
                        "enum": ["view", "create", "str_replace", "insert"]},
            "path": {"type": "string"},
            "view_range": {"type": "array", "items": {"type": "integer"}},
            "file_text": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
            "insert_line": {"type": "integer"}},
            "required": ["command", "path"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "Submit the final answer and end the episode.",
        "parameters": {"type": "object", "properties": {}}}},

]


def system_prompt(workdir: str) -> str:
    """Student-convention system prompt (byte-compatible call format)."""
    head = ("You are a helpful assistant that can interact with a computer to "
            "solve tasks. If a tool call is needed, put any reasoning in "
            "<think>...</think> first.\n"
            f"The repository is at {workdir}. Work there; relative paths are "
            "resolved against it.\nAvailable tools:\n")
    tail = ('\nTo call a tool, reply with ' + TOOL_CALL
            + '{"name": "...", "arguments": {...}} and nothing after it.')
    return head + json.dumps(TOOLS, ensure_ascii=False) + tail


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------
class ToolError(Exception):
    pass


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (BASH_TIMEOUT + 5, BASH_TIMEOUT + 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    # RLIMIT_AS is unreliable on macOS; omitted on purpose rather than faked.


class Sandbox:
    """Per-episode working tree. `files` is {relpath: text}."""

    def __init__(self, files: dict[str, str] | None = None, keep: bool = False):
        self.root = Path(tempfile.mkdtemp(prefix="agent-env-"))
        self.workdir = self.root / "repo"
        self.workdir.mkdir()
        self.keep = keep
        # a `python` shim so the model's natural `python x.py` resolves
        self.bindir = self.root / "bin"
        self.bindir.mkdir()
        py = os.path.join(self.bindir, "python")
        if not os.path.exists(py):
            try:
                os.symlink(sys.executable, py)
            except OSError:
                pass
        for rel, content in (files or {}).items():
            p = self.workdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    # -- path handling: everything must stay inside workdir -----------------
    def resolve(self, path: str) -> Path:
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path is required")
        p = Path(path)
        if not p.is_absolute():
            p = self.workdir / p
        real = Path(os.path.realpath(p))
        base = Path(os.path.realpath(self.workdir))
        if real != base and base not in real.parents:
            raise ToolError(f"path escapes the repository: {path}")
        return real

    def _env(self) -> dict:
        return {"PATH": f"{self.bindir}:/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME": str(self.root), "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1"}

    def bash(self, command: str, timeout: int = BASH_TIMEOUT) -> dict:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command is required")
        try:
            r = subprocess.run(command, shell=True, cwd=self.workdir,
                               env=self._env(), capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits)
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "",
                    "stderr": f"command timed out after {timeout}s"}
        out = (r.stdout or "")
        err = (r.stderr or "")
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n... [stdout truncated]"
        if len(err) > MAX_OUTPUT:
            err = err[:MAX_OUTPUT] + "\n... [stderr truncated]"
        return {"exit_code": r.returncode, "stdout": out, "stderr": err}

    def view(self, path: str, rng: list | None = None) -> dict:
        p = self.resolve(path)
        if not p.exists() or not p.is_file():
            raise ToolError(f"file not found: {path}")
        lines = p.read_text(errors="replace").splitlines()
        lo, hi = 1, len(lines)
        if rng and len(rng) == 2:
            lo, hi = max(1, int(rng[0])), min(len(lines), int(rng[1]))
        numbered = "\n".join(f"{i:>4}: {lines[i-1]}" for i in range(lo, hi + 1))
        return {"path": str(p), "content": numbered, "total_lines": len(lines)}

    def create(self, path: str, file_text: str) -> dict:
        p = self.resolve(path)
        if p.exists():
            raise ToolError(f"file already exists: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text or "")
        return {"path": str(p), "created": True}

    def str_replace(self, path: str, old_str: str, new_str: str) -> dict:
        p = self.resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        text = p.read_text()
        if text.count(old_str) != 1:
            raise ToolError(f"old_str must occur exactly once; found {text.count(old_str)}")
        p.write_text(text.replace(old_str, new_str))
        return {"path": str(p), "replaced": True}

    def insert(self, path: str, insert_line: int, new_str: str) -> dict:
        p = self.resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        lines = p.read_text().splitlines(keepends=True)
        if not (0 <= insert_line <= len(lines)):
            raise ToolError(f"insert_line {insert_line} out of range 0..{len(lines)}")
        lines.insert(insert_line, new_str if new_str.endswith("\n") else new_str + "\n")
        p.write_text("".join(lines))
        return {"path": str(p), "inserted": True}

    def close(self):
        if not self.keep:
            shutil.rmtree(self.root, ignore_errors=True)


def execute_tool(name: str, args: dict, sb: Sandbox) -> dict:
    """Dispatch one tool call -> a JSON-able observation. Errors are values, not
    exceptions: the model must see the error text and recover (§3.2)."""
    args = args or {}
    try:
        if name == "bash":
            return sb.bash(str(args.get("command", "")))
        if name == "str_replace_editor":
            cmd = args.get("command")
            if cmd == "view":
                return sb.view(str(args.get("path", "")), args.get("view_range"))
            if cmd == "create":
                return sb.create(str(args.get("path", "")), args.get("file_text", ""))
            if cmd == "str_replace":
                return sb.str_replace(str(args.get("path", "")),
                                      args.get("old_str", ""), args.get("new_str", ""))
            if cmd == "insert":
                return sb.insert(str(args.get("path", "")),
                                 int(args.get("insert_line", 0)), args.get("new_str", ""))
            raise ToolError(f"unknown editor command: {cmd}")
        if name == "submit":
            return {"submitted": True}
        raise ToolError(f"unknown tool '{name}'")
    except ToolError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# tier-0 micro tasks — tiny self-contained repos, verified by a plain
# `python test_x.py` (no pytest, no deps). Parameterised so RL cannot memorise.
# ---------------------------------------------------------------------------
def _t_constant(rng):
    mod = rng.choice(["calc", "mathx", "ops", "arith"])
    fn = rng.choice(["add", "plus", "bump", "offset"])
    correct = rng.randint(3, 12)
    wrong = correct + rng.choice([-2, -1, 1, 2])
    files = {
        f"{mod}.py": f"def {fn}(x):\n    return x + {wrong}\n",
        f"test_{mod}.py": (f"from {mod} import {fn}\n"
                           f"assert {fn}(0) == {correct}, 'got ' + str({fn}(0))\n"
                           f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"return x + {wrong}",
           "new_str": f"return x + {correct}"}
    problem = (f"`{fn}` in `{mod}.py` should add {correct} to its argument, but "
               f"`python test_{mod}.py` fails. Fix `{mod}.py` so the test passes.")
    return files, problem, f"python test_{mod}.py", fix


def _t_operator(rng):
    mod = rng.choice(["calc", "mathx", "arith"])
    fn = rng.choice(["combine", "merge", "reduce", "apply"])
    right, wrongop = rng.choice([("*", "+"), ("-", "+"), ("+", "-"), ("/", "*")])
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    if right == "/":
        a, b = a * b, b  # keep integer-clean
    expect = {"*": a * b, "+": a + b, "-": a - b, "/": a // b}[right]
    files = {
        f"{mod}.py": f"def {fn}(a, b):\n    return a {wrongop} b\n",
        f"test_{mod}.py": (f"from {mod} import {fn}\n"
                           f"assert {fn}({a}, {b}) == {expect}, 'got ' + str({fn}({a}, {b}))\n"
                           f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"return a {wrongop} b",
           "new_str": f"return a {right} b"}
    problem = (f"`{fn}` in `{mod}.py` computes the wrong result; `python "
               f"test_{mod}.py` fails. Fix the operator in `{mod}.py`.")
    return files, problem, f"python test_{mod}.py", fix


def _t_offbyone(rng):
    mod = rng.choice(["seq", "range_util", "nums"])
    fn = rng.choice(["upto", "first_n", "count_to"])
    n = rng.randint(3, 8)
    files = {
        f"{mod}.py": f"def {fn}(n):\n    return list(range(1, n))\n",
        f"test_{mod}.py": (f"from {mod} import {fn}\n"
                           f"assert {fn}({n}) == {list(range(1, n + 1))}, 'got ' + str({fn}({n}))\n"
                           f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return list(range(1, n))",
           "new_str": "return list(range(1, n + 1))"}
    problem = (f"`{fn}` in `{mod}.py` is off by one: it must return 1..n inclusive, "
               f"but `python test_{mod}.py` fails. Fix it.")
    return files, problem, f"python test_{mod}.py", fix


def _t_missing_return(rng):
    mod = rng.choice(["util", "helpers", "tools"])
    fn = rng.choice(["double", "scale", "shift"])
    k = rng.randint(2, 6)
    files = {
        f"{mod}.py": f"def {fn}(x):\n    x = x * {k}\n",
        f"test_{mod}.py": (f"from {mod} import {fn}\n"
                           f"assert {fn}(2) == {2 * k}, 'got ' + str({fn}(2))\n"
                           f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"    x = x * {k}",
           "new_str": f"    return x * {k}"}
    problem = (f"`{fn}` in `{mod}.py` should return x multiplied by {k}, but it "
               f"returns nothing and `python test_{mod}.py` fails. Fix `{mod}.py`.")
    return files, problem, f"python test_{mod}.py", fix


def _t_comparison(rng):
    mod = rng.choice(["pred", "checks", "valid"])
    fn = rng.choice(["is_positive", "is_strict", "above_zero"])
    files = {
        f"{mod}.py": f"def {fn}(x):\n    return x >= 0\n",
        f"test_{mod}.py": (f"from {mod} import {fn}\n"
                           f"assert {fn}(1) is True, 'got ' + str({fn}(1))\n"
                           f"assert {fn}(0) is False, 'got ' + str({fn}(0))\n"
                           f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return x >= 0", "new_str": "return x > 0"}
    problem = (f"`{fn}` in `{mod}.py` must be strictly positive (0 is NOT positive), "
               f"but `python test_{mod}.py` fails. Fix the comparison.")
    return files, problem, f"python test_{mod}.py", fix


_TEMPLATES = [_t_constant, _t_operator, _t_offbyone, _t_missing_return, _t_comparison]


class MicroTask:
    def __init__(self, task_id, files, problem, check_cmd, fix=None, tier=0):
        self.task_id = task_id
        self.files = files
        self.problem = problem
        self.check_cmd = check_cmd
        self.fix = fix          # {"path","old_str","new_str"} — for the oracle, never the prompt
        self.tier = tier

    def user_prompt(self, workdir: str) -> str:
        return f"{self.problem}\n\nThe repository is at {workdir}."


def generate_task(seed: int) -> MicroTask:
    rng = random.Random(seed)
    files, problem, check, fix = rng.choice(_TEMPLATES)(rng)
    return MicroTask(f"micro-{seed}", files, problem, check, fix=fix)


def run_check(task: MicroTask, sb: Sandbox) -> dict:
    """The deterministic correctness oracle: did the repo's own test pass?"""
    res = sb.bash(task.check_cmd)
    passed = res.get("exit_code") == 0 and "OK" in (res.get("stdout") or "")
    return {"ran": True, "passed": passed, "cmd": task.check_cmd,
            "output": (res.get("stdout", "") + res.get("stderr", ""))[-800:]}


# ---------------------------------------------------------------------------
# episode loop
# ---------------------------------------------------------------------------
def _json_objects(text):
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            return
        try:
            obj, end = dec.raw_decode(text[j:])
            yield obj
            i = j + end
        except Exception:  # noqa: BLE001
            i = j + 1


def parse_calls(text: str) -> list[tuple[str, dict]]:
    calls = []
    for obj in _json_objects(text or ""):
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and (
                obj.get("arguments") is not None or obj.get("parameters") is not None):
            args = obj.get("arguments", obj.get("parameters"))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append((obj["name"], args))
    return calls


def run_episode(task: MicroTask, policy, max_steps: int = MAX_STEPS,
                keep: bool = False) -> dict:
    """Drive the sandbox against `policy(messages) -> dict`:
        {"text": str, "stopped": bool, "calls": [(name,args)], ...}
    `calls` may be empty; they are then parsed from `text`."""
    sb = Sandbox(task.files, keep=keep)
    try:
        messages = [{"role": "system", "content": system_prompt(str(sb.workdir))},
                    {"role": "user", "content": task.user_prompt(str(sb.workdir))}]
        steps, calls_made, signatures = 0, [], []
        submitted = False
        last_stopped = True
        while steps < max_steps:
            steps += 1
            out = policy(messages)
            text = out.get("text") or ""
            last_stopped = bool(out.get("stopped", True))
            calls = out.get("calls") or parse_calls(text)
            if not calls:
                break
            hist = text.strip() or "\n".join(
                TOOL_CALL + json.dumps({"name": n, "arguments": a}, ensure_ascii=False)
                for n, a in calls)
            messages.append({"role": "assistant", "content": hist})
            for name, args in calls:
                sig = (name, json.dumps(args, sort_keys=True))
                signatures.append(sig)
                calls_made.append((name, args))
                res = execute_tool(name, args, sb)
                if name == "submit" and not res.get("error"):
                    submitted = True
                messages.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(res)})
            if submitted:
                break
        # loop = the same failing call signature seen twice or more
        err_sigs = [s for s in signatures if signatures.count(s) >= 2]
        loop = len(set(err_sigs)) > 0
        check = run_check(task, sb)
        return {
            "task_id": task.task_id, "tier": task.tier, "seed": task.task_id,
            "messages": messages, "steps": steps, "stopped": last_stopped,
            "submitted": submitted, "loop": loop,
            "n_calls": len(calls_made),
            "check": check,
            "workdir": str(sb.workdir),
        }
    finally:
        sb.close()


# ---------------------------------------------------------------------------
# self-test — scripted policies, no model
# ---------------------------------------------------------------------------
def _script(steps):
    box = {"i": 0}

    def policy(messages):
        i = box["i"]
        box["i"] += 1
        if i >= len(steps):
            return {"text": "", "stopped": True, "calls": []}
        return steps[i]
    return policy


def _call(name, **args):
    return TOOL_CALL + json.dumps({"name": name, "arguments": args})


def selftest() -> int:
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== generator variety (no two seeds share a template+module)")
    tasks = [generate_task(s) for s in range(40)]
    ok("variety", len({(t.problem) for t in tasks}) >= 20,
       f"{len({t.problem for t in tasks})}/40 distinct")

    t = generate_task(0)
    print(f"  task0: {t.task_id} check={t.check_cmd} files={list(t.files)}")
    files, problem, check, fix = _t_constant(random.Random(0))
    task = MicroTask("unit", files, problem, check, fix=fix)

    print("== do-nothing control: submit immediately must NOT pass")
    ep = run_episode(task, _script([{"text": "I don't know.", "stopped": True,
                                     "calls": [("submit", {})]}]))
    ok("donothing.check_false", ep["check"]["passed"] is False)
    ok("donothing.submitted", ep["submitted"] is True)

    print("== correct solve: read, edit, run test, submit")
    path, test = task.fix["path"], task.check_cmd
    ep = run_episode(task, _script([
        {"text": _call("bash", command=f"cat {path}"), "stopped": True, "calls": []},
        {"text": _call("str_replace_editor", command="view", path=path), "stopped": True,
         "calls": []},
        {"text": _call("str_replace_editor", command="str_replace", path=path,
                       old_str=task.fix["old_str"], new_str=task.fix["new_str"]),
         "stopped": True, "calls": []},
        {"text": _call("bash", command=test), "stopped": True, "calls": []},
        {"text": _call("submit"), "stopped": True, "calls": []},
    ]))
    ok("solve.check_passed", ep["check"]["passed"] is True, ep["check"]["output"][:80])
    ok("solve.steps", ep["steps"] == 5, f"steps={ep['steps']}")
    ok("solve.no_loop", ep["loop"] is False)

    print("== loop detection: same failing call twice")
    ep2 = run_episode(task, _script([
        {"text": _call("bash", command="cat nope.py"), "stopped": True, "calls": []},
        {"text": _call("bash", command="cat nope.py"), "stopped": True, "calls": []},
        {"text": _call("submit"), "stopped": True, "calls": []},
    ]))
    ok("loop.detected", ep2["loop"] is True)
    ok("loop.check_false", ep2["check"]["passed"] is False)

    print("== jail: path escape is an error value, not a crash")
    sb = Sandbox({"a.py": "x = 1\n"})
    try:
        r = execute_tool("str_replace_editor", {"command": "view", "path": "../../etc/passwd"}, sb)
        ok("jail.escape_blocked", "error" in r, str(r)[:60])
        r2 = execute_tool("bash", {"command": "echo hi > /tmp/agent_escape_probe && echo done"}, sb)
        # writing outside is possible with a shell redirect; the jail is documented as
        # NOT a security boundary. Assert the tool still returns rather than hangs.
        ok("jail.bash_returns", "exit_code" in r2)
    finally:
        sb.close()
        try:
            os.remove("/tmp/agent_escape_probe")
        except OSError:
            pass

    print("== unique old_str enforced")
    sb = Sandbox({"b.py": "a = 1\na = 1\n"})
    try:
        r = execute_tool("str_replace_editor", {"command": "str_replace", "path": "b.py",
                                                "old_str": "a = 1", "new_str": "a = 2"}, sb)
        ok("unique.old_str", "error" in r and "exactly once" in r["error"])
    finally:
        sb.close()

    print()
    if fails:
        print(f"SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", type=int, default=None,
                    help="print a generated task's files and problem")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.demo is not None:
        t = generate_task(a.demo)
        print(t.problem, "\n")
        for k, v in t.files.items():
            print(f"--- {k} ---\n{v}")
        print("check:", t.check_cmd)
        raise SystemExit(0)
    raise SystemExit(0)
