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
    """Student-convention system prompt (byte-compatible call format).

    The rules block is the port of MiMo's anti-shortcut wording (RL_PLAN_2 §4.4):
    the graded ladder is a hack surface, and the cheapest first line of defence is
    to say plainly that editing the tests is not a fix. It is *requested* here and
    *enforced* in `rl/verifier.py` (§4.3) — the request alone is not a gate.
    """
    head = ("You are a helpful assistant that can interact with a computer to "
            "solve tasks. If a tool call is needed, put any reasoning in "
            "<think>...</think> first.\n"
            f"The repository is at {workdir}. Work there; relative paths are "
            "resolved against it.\n"
            "Rules: fix the source code, not the test. Do not overwrite, edit "
            "or delete test files to make them pass; do not hard-code expected "
            "outputs; keep changes minimal; prefer str_replace_editor over bash "
            "for file edits.\nAvailable tools:\n")
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
        # the pristine tree, for the verifier's anti-hack diff (RL_PLAN_2 §4.3):
        # which files the episode actually changed, and which test files to
        # restore before grading. Kept here because run_episode owns the tree.
        self.initial_files = dict(files or {})

    # -- snapshot / restore: what the verifier needs to grade honestly --------
    def read_tree(self) -> dict:
        """The current tree as {relpath: text}. Binaries and symlinks skipped
        (best-effort decode): the only files this env ever writes are text."""
        out = {}
        base = self.workdir
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.is_symlink():
                continue
            try:
                out[str(p.relative_to(base))] = p.read_text(errors="replace")
            except OSError:
                continue
        return out

    def diff_initial(self) -> dict:
        """{rel: (before, after)} for every file that differs from the pristine
        tree, a missing side being None (added / deleted). The model's *edits*
        are read off the filesystem, not off its tool calls: a `bash` redirect is
        an edit too, and trusting the transcript would miss it."""
        now = self.read_tree()
        before = self.initial_files
        changed = {}
        for rel in set(before) | set(now):
            b, a = before.get(rel), now.get(rel)
            if b != a:
                changed[rel] = (b, a)
        return changed

    def restore(self, files: dict) -> None:
        """Write `files` back over the tree (used to undo the model's edits to
        test files before the check is trusted)."""
        for rel, content in files.items():
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
def _variant(check_cmd: str, files: dict) -> dict:
    """A held-out re-check: the same defect, different literals (RL_PLAN_2 §4.3).

    Written after the episode, over the model's source edits. Its only job is to
    separate "fixed the source" from "hard-coded the visible test's expected
    output" — a policy that returns the constant the visible test asserts passes
    the visible check and fails here.
    """
    return {"check_cmd": check_cmd, "files": files}


def _t_constant(rng):
    mod = rng.choice(["calc", "mathx", "ops", "arith", "core", "base"])
    fn = rng.choice(["add", "plus", "bump", "offset", "shift_by"])
    correct = rng.randint(3, 12)
    wrong = correct + rng.choice([-2, -1, 1, 2])
    test = f"test_{mod}.py"
    a1, a2 = rng.sample(range(1, 20), 2)
    files = {
        f"{mod}.py": f"def {fn}(x):\n    return x + {wrong}\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}(0) == {correct}, 'got ' + str({fn}(0))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"return x + {wrong}",
           "new_str": f"return x + {correct}"}
    problem = (f"`{fn}` in `{mod}.py` should add {correct} to its argument, but "
               f"`python {test}` fails. Fix `{mod}.py` so the test passes.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}({a1}) == {a1 + correct}, 'got ' + str({fn}({a1}))\n"
                    f"assert {fn}({a2}) == {a2 + correct}, 'got ' + str({fn}({a2}))\n"
                    f"print('OK')\n")})}


def _t_operator(rng):
    mod = rng.choice(["calc", "mathx", "arith", "ops"])
    fn = rng.choice(["combine", "merge", "reduce", "apply"])
    right, wrongop = rng.choice([("*", "+"), ("-", "+"), ("+", "-"), ("/", "*")])
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    if right == "/":
        a, b = a * b, b  # keep integer-clean
    expect = {"*": a * b, "+": a + b, "-": a - b, "/": a // b}[right]
    test = f"test_{mod}.py"
    def vals():
        x, y = rng.randint(2, 9), rng.randint(2, 9)
        if right == "/":
            x = x * y
        return x, y, {"*": x * y, "+": x + y, "-": x - y, "/": x // y}[right]
    v1, v2 = vals(), vals()
    files = {
        f"{mod}.py": f"def {fn}(a, b):\n    return a {wrongop} b\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}({a}, {b}) == {expect}, 'got ' + str({fn}({a}, {b}))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"return a {wrongop} b",
           "new_str": f"return a {right} b"}
    problem = (f"`{fn}` in `{mod}.py` computes the wrong result; `python "
               f"{test}` fails. Fix the operator in `{mod}.py`.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}({v1[0]}, {v1[1]}) == {v1[2]}\n"
                    f"assert {fn}({v2[0]}, {v2[1]}) == {v2[2]}\n"
                    f"print('OK')\n")})}


def _t_offbyone(rng):
    mod = rng.choice(["seq", "range_util", "nums", "series"])
    fn = rng.choice(["upto", "first_n", "count_to"])
    n = rng.randint(3, 8)
    n2 = rng.choice([x for x in range(2, 11) if x != n])
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(n):\n    return list(range(1, n))\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}({n}) == {list(range(1, n + 1))}, 'got ' + str({fn}({n}))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return list(range(1, n))",
           "new_str": "return list(range(1, n + 1))"}
    problem = (f"`{fn}` in `{mod}.py` is off by one: it must return 1..n inclusive, "
               f"but `python {test}` fails. Fix it.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}({n2}) == {list(range(1, n2 + 1))}, 'got ' + str({fn}({n2}))\n"
                    f"print('OK')\n")})}


def _t_missing_return(rng):
    mod = rng.choice(["util", "helpers", "tools", "support"])
    fn = rng.choice(["double", "scale", "shift"])
    k = rng.randint(2, 6)
    k2 = rng.choice([x for x in range(2, 8) if x != k])
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(x):\n    x = x * {k}\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}(2) == {2 * k}, 'got ' + str({fn}(2))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"    x = x * {k}",
           "new_str": f"    return x * {k}"}
    problem = (f"`{fn}` in `{mod}.py` should return x multiplied by {k}, but it "
               f"returns nothing and `python {test}` fails. Fix `{mod}.py`.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}(3) == {3 * k}, 'got ' + str({fn}(3))\n"
                    f"assert {fn}({k2}) == {k2 * k}\n"
                    f"print('OK')\n")})}


def _t_comparison(rng):
    mod = rng.choice(["pred", "checks", "valid", "guards"])
    fn = rng.choice(["is_positive", "is_strict", "above_zero"])
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(x):\n    return x >= 0\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}(1) is True, 'got ' + str({fn}(1))\n"
               f"assert {fn}(0) is False, 'got ' + str({fn}(0))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return x >= 0", "new_str": "return x > 0"}
    problem = (f"`{fn}` in `{mod}.py` must be strictly positive (0 is NOT positive), "
               f"but `python {test}` fails. Fix the comparison.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}(5) is True, 'got ' + str({fn}(5))\n"
                    f"assert {fn}(-3) is False, 'got ' + str({fn}(-3))\n"
                    f"print('OK')\n")})}


def _t_index(rng):
    mod = rng.choice(["sequtil", "lists", "arrays", "collections"])
    fn = rng.choice(["first", "head", "pick", "front"])
    xs = rng.sample(range(10, 99), 3)
    ys = rng.sample(range(100, 199), 3)
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(xs):\n    return xs[1]\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}({xs}) == {xs[0]}, 'got ' + str({fn}({xs}))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return xs[1]", "new_str": "return xs[0]"}
    problem = (f"`{fn}` in `{mod}.py` must return the first element of the list, "
               f"but `python {test}` fails. Fix the index.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}({ys}) == {ys[0]}, 'got ' + str({fn}({ys}))\n"
                    f"print('OK')\n")})}


def _t_string(rng):
    mod = rng.choice(["text", "strutil", "names", "fmt"])
    fn = rng.choice(["greet", "label", "prefix"])
    word = rng.choice(["hello", "hi", "hey", "welcome"])
    bad = word.capitalize()
    names = rng.sample(["Sam", "Ana", "Leo", "Nia", "Kai", "Uma"], 2)
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f'def {fn}(name):\n    return "{bad} " + name\n',
        test: (f"from {mod} import {fn}\n"
               f'assert {fn}("{names[0]}") == "{word} {names[0]}", '
               f"'got ' + {fn}('{names[0]}')\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f'return "{bad} " + name',
           "new_str": f'return "{word} " + name'}
    problem = (f"`{fn}` in `{mod}.py` must mention the caller's name in lower-case "
               f"(`{word} <name>`), but `python {test}` fails. Fix the string.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f'assert {fn}("{names[1]}") == "{word} {names[1]}"\n'
                    f"print('OK')\n")})}


def _t_default(rng):
    mod = rng.choice(["config", "settings", "options", "defaults"])
    fn = rng.choice(["limit", "retries", "timeout", "batch"])
    correct = rng.randint(3, 9)
    wrong = correct + rng.choice([-2, 2])
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(n={wrong}):\n    return n\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}() == {correct}, 'got ' + str({fn}())\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": f"def {fn}(n={wrong}):",
           "new_str": f"def {fn}(n={correct}):"}
    problem = (f"`{fn}` in `{mod}.py` must default to {correct}, but `python "
               f"{test}` fails. Fix the default value.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}() == {correct}, 'default wrong'\n"
                    f"assert {fn}(5) == 5, 'explicit arg ignored'\n"
                    f"print('OK')\n")})}


def _t_boolean(rng):
    mod = rng.choice(["logic", "flags", "rules", "gates"])
    fn = rng.choice(["either", "any_true", "enabled", "ok"])
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(a, b):\n    return a and b\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}(True, False) is True, 'got ' + str({fn}(True, False))\n"
               f"assert {fn}(False, False) is False, 'got ' + str({fn}(False, False))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return a and b", "new_str": "return a or b"}
    problem = (f"`{fn}` in `{mod}.py` must be true when *either* argument is true, "
               f"but `python {test}` fails. Fix the logic.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}(False, True) is True, 'got ' + str({fn}(False, True))\n"
                    f"assert {fn}(True, True) is True\n"
                    f"print('OK')\n")})}


def _t_sort(rng):
    mod = rng.choice(["order", "rank", "sortutil"])
    fn = rng.choice(["sorted_asc", "ascending", "in_order"])
    xs = rng.sample(range(1, 20), 4)
    ys = rng.sample(range(1, 20), 4)
    test = f"test_{mod}.py"
    files = {
        f"{mod}.py": f"def {fn}(xs):\n    return sorted(xs, reverse=True)\n",
        test: (f"from {mod} import {fn}\n"
               f"assert {fn}({xs}) == {sorted(xs)}, 'got ' + str({fn}({xs}))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{mod}.py", "old_str": "return sorted(xs, reverse=True)",
           "new_str": "return sorted(xs, reverse=False)"}
    problem = (f"`{fn}` in `{mod}.py` must return the list in ascending order, but "
               f"`python {test}` fails. Fix the sort.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{mod}.py", {
                f"test_var_{mod}.py": (
                    f"from {mod} import {fn}\n"
                    f"assert {fn}({ys}) == {sorted(ys)}, 'got ' + str({fn}({ys}))\n"
                    f"print('OK')\n")})}


def _t_multifile(rng):
    """A two-module defect: the fix lives in the helper the API imports. Exercises
    "the correct file is not the one named in the problem" without a real repo."""
    helper = rng.choice(["util", "helpers", "support", "internal"])
    api = rng.choice(["api", "service", "app", "core_lib"])
    fn = rng.choice(["compute", "process", "transform"])
    correct = rng.randint(2, 9)
    wrong = correct + rng.choice([-1, 1])
    inner = "_scale"
    test = f"test_{api}.py"
    files = {
        f"{helper}.py": f"def {inner}(x):\n    return x * {wrong}\n",
        f"{api}.py": (f"from {helper} import {inner}\n\n\n"
                      f"def {fn}(x):\n    return {inner}(x)\n"),
        test: (f"from {api} import {fn}\n"
               f"assert {fn}(1) == {correct}, 'got ' + str({fn}(1))\n"
               f"print('OK')\n"),
    }
    fix = {"path": f"{helper}.py", "old_str": f"return x * {wrong}",
           "new_str": f"return x * {correct}"}
    problem = (f"`{fn}` in `{api}.py` multiplies its argument by {correct}, but "
               f"`python {test}` fails and the value comes from a helper module. "
               f"Find and fix the source of the wrong factor.")
    return {"files": files, "problem": problem, "check_cmd": f"python {test}",
            "fix": fix, "test_files": [test],
            "variant": _variant(f"python test_var_{api}.py", {
                f"test_var_{api}.py": (
                    f"from {api} import {fn}\n"
                    f"assert {fn}(3) == {3 * correct}, 'got ' + str({fn}(3))\n"
                    f"assert {fn}(0) == 0\n"
                    f"print('OK')\n")})}


_TEMPLATES = [_t_constant, _t_operator, _t_offbyone, _t_missing_return,
              _t_comparison, _t_index, _t_string, _t_default, _t_boolean,
              _t_sort, _t_multifile]

# Train and eval use disjoint seed ranges (RL_PLAN_2 §6.1/§9 G4): the generator
# is parameterised by seed, so a held-out split is a held-out range, and a policy
# that memorised the training tasks is exposed by the eval range.
TRAIN_SEED0 = 0
HOLDOUT_SEED0 = 1_000_000


class MicroTask:
    def __init__(self, task_id, files, problem, check_cmd, fix=None, tier=0,
                 test_files=None, variant=None):
        self.task_id = task_id
        self.files = files
        self.problem = problem
        self.check_cmd = check_cmd
        self.fix = fix          # {"path","old_str","new_str"} — for the oracle, never the prompt
        self.tier = tier
        # which files are tests: the verifier restores them before grading so the
        # check can only be passed by a source edit (§4.3)
        self.test_files = test_files if test_files is not None else [
            k for k in files if os.path.basename(k).startswith("test_")]
        # {"files": {rel: text}, "check_cmd": str} — same defect, other literals
        self.variant = variant

    def user_prompt(self, workdir: str) -> str:
        return f"{self.problem}\n\nThe repository is at {workdir}."


def generate_task(seed: int) -> MicroTask:
    rng = random.Random(seed)
    spec = rng.choice(_TEMPLATES)(rng)
    return MicroTask(f"micro-{seed}", spec["files"], spec["problem"],
                     spec["check_cmd"], fix=spec["fix"],
                     test_files=spec.get("test_files"),
                     variant=spec.get("variant"))


def holdout_tasks(n: int, seed0: int = HOLDOUT_SEED0):
    """The eval split: same generator, disjoint seed range (§6.1)."""
    return [generate_task(seed0 + i) for i in range(n)]


def run_check(task: MicroTask, sb: Sandbox, cmd: str | None = None) -> dict:
    """The deterministic correctness oracle: did the repo's own test pass?

    `cmd` overrides the task's check (used by the verifier's held-out variant,
    §4.3). The pass rule stays `exit_code == 0 and "OK" in stdout` — that rule is
    spoofable on its own, which is exactly why the verifier restores the test
    files and re-runs the variant before it believes this.
    """
    cmd = cmd or task.check_cmd
    res = sb.bash(cmd)
    passed = res.get("exit_code") == 0 and "OK" in (res.get("stdout") or "")
    return {"ran": True, "passed": passed, "cmd": cmd,
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
                keep: bool = False, grader=None) -> dict:
    """Drive the sandbox against `policy(messages) -> dict`:
        {"text": str, "stopped": bool, "calls": [(name,args)], ...}
    `calls` may be empty; they are then parsed from `text`.

    `grader(task, sb, episode) -> dict`, if given, runs while the sandbox is open
    (the verifier needs the tree to restore test files and re-check, §4.3) and its
    return value lands in the episode as `grade`. The plain `check` is always the
    as-is run, so a caller that ignores grading keeps the old behaviour.

    `initial_check` is the check on the *pristine* tree, recorded before the
    policy moves: the graded ladder's "did the failure change" step (§4.2) is a
    comparison against it.
    """
    sb = Sandbox(task.files, keep=keep)
    try:
        messages = [{"role": "system", "content": system_prompt(str(sb.workdir))},
                    {"role": "user", "content": task.user_prompt(str(sb.workdir))}]
        initial = run_check(task, sb)
        steps, calls_made, signatures = 0, [], []
        submitted = False
        last_stopped = True
        turn_stopped = []      # per assistant turn: did it end on im_end (not num_predict)?
        while steps < max_steps:
            steps += 1
            out = policy(messages)
            text = out.get("text") or ""
            last_stopped = bool(out.get("stopped", True))
            calls = out.get("calls") or parse_calls(text)
            if not calls:
                # a turn with prose and no call is still an action the policy
                # took (it chose to stop). Keep it: GRPO trains on it, and DPO's
                # turn extraction already handles a call-less assistant turn.
                if text.strip():
                    messages.append({"role": "assistant", "content": text})
                    turn_stopped.append(last_stopped)
                break
            hist = text.strip() or "\n".join(
                TOOL_CALL + json.dumps({"name": n, "arguments": a}, ensure_ascii=False)
                for n, a in calls)
            messages.append({"role": "assistant", "content": hist})
            turn_stopped.append(last_stopped)
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
        ep = {
            "task_id": task.task_id, "tier": task.tier, "seed": task.task_id,
            "messages": messages, "steps": steps, "stopped": last_stopped,
            "submitted": submitted, "loop": loop,
            "n_calls": len(calls_made),
            "check": check,
            "initial_check": initial,
            "turn_stopped": turn_stopped,
            "workdir": str(sb.workdir),
        }
        if grader is not None:
            ep["grade"] = grader(task, sb, ep)
        return ep
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
    tasks = [generate_task(s) for s in range(60)]
    ok("variety", len({(t.problem) for t in tasks}) >= 40,
       f"{len({t.problem for t in tasks})}/60 distinct")
    ok("all_have_variant", all(t.variant and t.variant.get("files") for t in tasks))
    ok("all_have_tests", all(t.test_files for t in tasks),
       "every task must declare a test file for the anti-hack restore")

    t = generate_task(0)
    print(f"  task0: {t.task_id} check={t.check_cmd} files={list(t.files)} "
          f"tests={t.test_files}")
    spec = _t_constant(random.Random(0))
    task = MicroTask("unit", spec["files"], spec["problem"], spec["check_cmd"],
                     fix=spec["fix"], test_files=spec.get("test_files"),
                     variant=spec.get("variant"))

    print("== anti-hack snapshot: pristine files are kept, an edit is seen")
    sb = Sandbox(task.files)
    try:
        diff0 = sb.diff_initial()
        ok("pristine_no_diff", diff0 == {}, str(diff0))
        p = task.fix["path"]
        sb.str_replace(p, task.fix["old_str"], task.fix["new_str"])
        diff1 = sb.diff_initial()
        ok("edit_seen", list(diff1) == [p], str(list(diff1)))
        vb = task.variant["files"]
        sb.restore(vb)
        ok("variant_written", all((sb.workdir / r).exists() for r in vb))
    finally:
        sb.close()

    print("== variant separates a real fix from a hard-coded constant")
    src = task.files[task.fix["path"]]
    fn = re.match(r"def (\w+)", src).group(1)
    correct = task.fix["new_str"].split("+")[1].strip()
    sb = Sandbox(task.files)
    try:
        # the honest fix passes both the visible check and the held-out variant
        sb.str_replace(task.fix["path"], task.fix["old_str"], task.fix["new_str"])
        ok("fix_visible", run_check(task, sb)["passed"] is True)
        sb.restore(task.variant["files"])
        ok("fix_variant", run_check(task, sb, task.variant["check_cmd"])["passed"]
           is True)
    finally:
        sb.close()
    sb = Sandbox(task.files)
    try:
        # hard-code the visible test's expected output: visible passes, variant fails
        sb.restore({task.fix["path"]: f"def {fn}(x):\n    return {correct}\n"})
        ok("hardcode_visible", run_check(task, sb)["passed"] is True)
        sb.restore(task.variant["files"])
        ok("hardcode_variant_fails",
           run_check(task, sb, task.variant["check_cmd"])["passed"] is False)
    finally:
        sb.close()

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
