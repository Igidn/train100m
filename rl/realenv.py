#!/usr/bin/env python3
"""Docker-free real-repo environment — MiMo's shape, run natively.

What the MiMo code tasks actually are (verified 2026-10-20 against
`XiaomiMiMo/verl@mimo-oss/recipes/code/mimoagent_runner.py`):

    environment = make_dataset_env(instance, **environment_config)
    environment.setup_environment()                       # per-instance image
    reward, test_output, _ = environment.calculate_reward()   # SWE-bench grader

Grading is the SWE-bench harness over a prebuilt per-instance image whose
`/testbed` is a clean git checkout of the *base* commit with dependencies
already installed (`format-code-task-001457` is a 445 MB single-layer image:
`/testbed` + `node_modules` + `.git`, and no eval script). The images are
extractable through the registry API, but the per-instance eval spec is
server-side, and Kaggle has no docker daemon.

So this module rebuilds the same semantics natively:

  * the tree is a real `git clone` at the instance's base commit, with the
    repo's own install command run once (`npm ci`, `pip install -e .`);
  * the action space is `agent_env.TOOLS` unchanged, so the student sees the
    same bytes it has seen in every round so far;
  * grading is SWE-bench's rule, not a shape of our own: the **hidden** test
    files are restored from the fix commit before the run, so a model that
    edits the tests cannot grade itself, and pass means the fix commit's tests
    pass and its untested neighbours still do.

`RepoSandbox` mirrors `agent_env.Sandbox`'s interface because `run_episode`
drives whichever sandbox it is handed. The one real difference is that
`diff_initial()` reads git instead of diffing an in-memory snapshot: a real
tree is 30k files (`node_modules`), so per-turn snapshots are off the table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_env import MAX_OUTPUT  # noqa: E402

# A real test command is not a 20 s `python test_x.py`: jest on a small repo is
# seconds, `npm ci` is a minute, and a first `pytest` collection can be longer.
BASH_TIMEOUT = 120
INSTALL_TIMEOUT = 1800
TEST_TIMEOUT = 900


class RepoError(Exception):
    pass


def _run(cmd, cwd=None, timeout=BASH_TIMEOUT, env=None, check=False):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RepoError(f"{cmd!r} failed ({r.returncode}): {(r.stderr or r.stdout)[-400:]}")
    return r


def _git(cwd, *args, check=True) -> str:
    r = _run(["git", *args], cwd=cwd, check=check)
    return (r.stdout or "")


# ---------------------------------------------------------------------------
# instance
# ---------------------------------------------------------------------------
class RepoInstance:
    """One real-repo task. Everything the env needs, nothing it does not."""

    def __init__(self, instance_id, repo, base_commit, fix_commit, problem,
                 test_files, source_files, check_cmd, install_cmd=None,
                 lang="js", fail_to_pass=None, image=None, meta=None):
        self.instance_id = instance_id
        self.repo = repo                     # clone url (or a local path)
        self.base_commit = base_commit
        self.fix_commit = fix_commit
        self.problem = problem
        self.test_files = list(test_files)
        self.source_files = list(source_files)
        self.check_cmd = check_cmd
        self.install_cmd = install_cmd
        self.lang = lang
        self.fail_to_pass = list(fail_to_pass or [])
        self.image = image                   # the MiMo image tag, when known
        self.meta = dict(meta or {})

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in (
            "instance_id", "repo", "base_commit", "fix_commit", "problem",
            "test_files", "source_files", "check_cmd", "install_cmd", "lang",
            "fail_to_pass", "image", "meta")}

    @classmethod
    def from_json(cls, d) -> "RepoInstance":
        return cls(**d)

    def __repr__(self):
        return (f"<{self.instance_id} {self.repo.rsplit('/', 1)[-1]}@"
                f"{self.base_commit[:8]} tests={len(self.test_files)}>")


def load_instances(path) -> list:
    return [RepoInstance.from_json(json.loads(l)) for l in open(path) if l.strip()]


# ---------------------------------------------------------------------------
# working tree: clone + install, once per instance
# ---------------------------------------------------------------------------
def default_root() -> str:
    root = os.environ.get("REALENV_ROOT")
    if root:
        return root
    return os.path.join(tempfile.gettempdir(), "realenv")


def _marker(inst, dest) -> Path:
    """Where `prepare` records that a tree is installed — *outside* the tree.

    It used to live at `dest/.realenv.json`, which `reset_tree`'s `git clean -fd`
    deletes: every episode then paid for a fresh `npm install`. On the scale run
    that was three installs per instance (gate, student, gold) instead of one,
    which is most of why 16 vault-action instances took 1.7 h.
    """
    return dest.parent / f".{dest.name}.realenv.json"


def prepare(inst: RepoInstance, root=None, force=False, quiet=False) -> Path:
    """Materialise `inst` at `root/<instance_id>`: clone, checkout, install.

    Idempotent: the marker records the commit and the install command, so a
    rerun (or a resumed Kaggle session) skips the expensive part. `keep` lives
    on the sandbox; the tree itself is meant to be reused across episodes.
    """
    root = Path(root or default_root())
    dest = root / inst.instance_id
    marker = _marker(inst, dest)
    if marker.exists() and dest.is_dir() and not force:
        state = json.loads(marker.read_text())
        if state.get("base_commit") == inst.base_commit and state.get("installed") == bool(inst.install_cmd):
            return dest

    marker.unlink(missing_ok=True)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    src = inst.repo
    if os.path.isdir(src):                      # local clone (offline / tests)
        _run(["git", "clone", "--quiet", src, str(dest)], check=True)
    else:
        _run(["git", "clone", "--quiet", src, str(dest)], check=True, timeout=INSTALL_TIMEOUT)
    _git(dest, "checkout", "--quiet", inst.base_commit)
    _git(dest, "config", "user.email", "env@tinyballs.local")
    _git(dest, "config", "user.name", "tinyballs env")

    installed = False
    if inst.install_cmd:
        cmd = inst.install_cmd
        # `npm ci` requires a lockfile *at this commit*: repos that added one
        # later would otherwise fail to install at an older base.
        if cmd.startswith("npm ci") and not (dest / "package-lock.json").exists():
            cmd = cmd.replace("npm ci", "npm install", 1)
        cmd = f"{cmd} --no-audit --no-fund" if cmd.startswith("npm") else cmd
        if not quiet:
            print(f"  [install] {inst.instance_id}: {cmd}", flush=True)
        r = _run(cmd, cwd=dest, timeout=INSTALL_TIMEOUT)
        if r.returncode != 0 and cmd.startswith("npm ci"):
            # A 2019 lockfile is not "in sync" with its own package.json by npm
            # 11's rules, and `ci` refuses where npm 6 installed happily. Same
            # tree, different resolver: fall back rather than drop the instance.
            cmd = cmd.replace("npm ci", "npm install", 1)
            print(f"  [install] {inst.instance_id}: ci refused, retrying `{cmd}`",
                  flush=True)
            r = _run(cmd, cwd=dest, timeout=INSTALL_TIMEOUT)
        if r.returncode != 0 and (dest / "package-lock.json").exists():
            # npm 10 cannot even read a lockfile v1 in some trees ("Cannot read
            # properties of null (reading 'edgesOut')" — hono). The lockfile is
            # the input it chokes on, so it goes; `--no-package-lock` keeps a
            # fresh one from appearing, and the deletion is committed below so
            # the tree the model sees is still self-consistent.
            print(f"  [install] {inst.instance_id}: lockfile unreadable, "
                  f"regenerating", flush=True)
            (dest / "package-lock.json").unlink()
            cmd = cmd.replace("--no-audit", "--no-package-lock --no-audit", 1) \
                if "--no-package-lock" not in cmd else cmd
            r = _run(cmd, cwd=dest, timeout=INSTALL_TIMEOUT)
        if r.returncode != 0:
            raise RepoError(f"install failed for {inst.instance_id}: "
                            f"{(r.stderr or r.stdout)[-600:]}")
        installed = True
        # `npm install` can rewrite a *tracked* lockfile. Commit it so the
        # episode's diff baseline is "base + env": otherwise every episode —
        # including a do-nothing one — reports a lockfile edit the model never
        # made, and the ladder would read it as progress.
        if _git(dest, "status", "--porcelain").strip():
            _git(dest, "add", "-A")          # ignored paths (node_modules) stay out
            if _git(dest, "diff", "--cached", "--name-only").strip():
                _git(dest, "commit", "-q", "-m", "env: install")

    marker.write_text(json.dumps(
        {"instance_id": inst.instance_id, "base_commit": inst.base_commit,
         "installed": installed}))
    return dest


def reset_tree(repo_dir) -> None:
    """Back to the prepared baseline: index *and* worktree, then junk removed.

    `reset --hard`, not `checkout -- .`: installing a hidden test or the gold is
    a *staged* checkout (`git checkout <commit> -- <path>`), and `checkout -- .`
    restores the worktree from that index. Using it here kept the gold in the
    tree and was why the first gate run reported `pass_after=False` while a
    manual run passed — the reset was a no-op after the gold went in.

    `git clean` without `-x` keeps ignored paths (node_modules, .venv) — the
    whole point of installing once.
    """
    _git(repo_dir, "reset", "--hard", "--quiet", "HEAD")
    _git(repo_dir, "clean", "-qfd", "-e", "node_modules", "-e", ".venv")
    for junk in (".jest-out.json",):
        p = Path(repo_dir) / junk
        if p.exists():
            p.unlink()


def install_hidden_tests(inst: RepoInstance, repo_dir) -> None:
    """SWE-bench's anti-hack step: the fix commit's test files, and nothing else.

    Restoring from the fix commit (rather than `git apply`-ing a stored patch)
    is the same bytes and cannot fail on context drift: every path here is a
    path the fix commit itself changed, so `git checkout <fix> -- <tests>` is
    the post-fix file by construction.

    Deliberately does **not** reset the rest of the tree: this runs immediately
    before grading, and the model's source edits are the thing being graded.
    Callers that want a clean slate (the gate, between its two phases) ask for
    `reset_tree` themselves.
    """
    _git(repo_dir, "checkout", "--quiet", inst.fix_commit, "--", *inst.test_files)


def apply_gold(inst: RepoInstance, repo_dir) -> None:
    """The reference solution: the fix commit's non-test files."""
    _git(repo_dir, "checkout", "--quiet", inst.fix_commit, "--", *inst.source_files)


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------
class RepoSandbox:
    """A real worktree behind `agent_env.execute_tool`'s five calls.

    Unlike `agent_env.Sandbox` this does not own the tree: it is handed a
    prepared repo and must leave it usable across episodes (`close` is a no-op
    unless `ephemeral`). The base commit is the diff baseline, so "what did the
    model change" is a git question, not a snapshot question.
    """

    def __init__(self, workdir, keep=True, path_prefixes=(), ephemeral=False):
        self.workdir = Path(workdir).resolve()
        self.keep = keep
        self.ephemeral = ephemeral
        self.path_prefixes = [str(p) for p in path_prefixes]
        if not (self.workdir / ".git").exists():
            raise RepoError(f"not a git worktree: {self.workdir}")
        self.base_commit = self._git("rev-parse", "HEAD").strip()
        # pristine test text matters only for the tools' own `view`; grading
        # reads the fix commit, not this.
        self.initial_files: dict = {}

    # -- git helpers ---------------------------------------------------------
    def _git(self, *args, check=True) -> str:
        return _git(self.workdir, *args, check=check)

    def changed(self) -> dict:
        """{relpath: status} for everything the model touched, from git.

        `--porcelain` reports tracked modifications and untracked additions in
        one pass. Renames arrive as `R  old -> new`; the two ends are kept.
        """
        out = self._git("status", "--porcelain=v1", "--untracked-files=all", "--", ".")
        changed = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            status, _, path = line[:2], line[2:], line[3:]
            if "->" in path:                      # rename: count both sides
                old, new = path.split(" -> ", 1)
                changed[old.strip().strip('"')] = "D"
                changed[new.strip().strip('"')] = "A"
                continue
            p = path.strip().strip('"')
            if p:
                changed[p] = status.strip() or "?"
        return changed

    def read_tree(self) -> dict:
        """Text of the *changed* files only (see module docstring)."""
        return {rel: after for rel, (_b, after) in self.diff_initial().items()
                if after is not None}

    def diff_initial(self) -> dict:
        """{rel: (before, after)} against the base commit; a missing side is None."""
        changed = {}
        for rel, status in self.changed().items():
            p = self.workdir / rel
            try:
                after = p.read_text(errors="replace") if p.is_file() else None
            except OSError:
                after = None
            before = None
            if status != "A":
                r = _run(["git", "show", f"{self.base_commit}:{rel}"],
                         cwd=self.workdir, check=False)
                before = r.stdout if r.returncode == 0 else None
            if before != after:
                changed[rel] = (before, after)
        return changed

    def restore(self, files: dict) -> None:
        for rel, content in (files or {}).items():
            p = self.workdir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)

    # -- path handling (same contract as agent_env.Sandbox) ------------------
    def resolve(self, path: str) -> Path:
        from agent_env import ToolError
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
        """PATH keeps node/npm/pytest reachable — a repo env is not a toy env.

        `node_modules/.bin` first: `npx jest` then resolves the repo's own
        version rather than whatever the host has.
        """
        local_bin = self.workdir / "node_modules" / ".bin"
        parts = [str(local_bin), *self.path_prefixes,
                 os.environ.get("PATH", "/usr/bin:/bin")]
        env = dict(os.environ)
        env.update({"PATH": os.pathsep.join(p for p in parts if p),
                    "HOME": str(self.workdir.parent),
                    "CI": "1",
                    "GIT_TERMINAL_PROMPT": "0",
                    "PYTHONDONTWRITEBYTECODE": "1"})
        return env

    def bash(self, command: str, timeout: int = BASH_TIMEOUT) -> dict:
        from agent_env import ToolError
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command is required")
        try:
            r = subprocess.run(command, shell=True, cwd=self.workdir,
                               env=self._env(), capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "",
                    "stderr": f"command timed out after {timeout}s"}
        out = r.stdout or ""
        err = r.stderr or ""
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n... [stdout truncated]"
        if len(err) > MAX_OUTPUT:
            err = err[:MAX_OUTPUT] + "\n... [stderr truncated]"
        return {"exit_code": r.returncode, "stdout": out, "stderr": err}

    # -- the editor, byte-identical semantics to agent_env.Sandbox -----------
    def view(self, path: str, rng: list | None = None) -> dict:
        from agent_env import ToolError
        p = self.resolve(path)
        if not p.exists() or not p.is_file():
            raise ToolError(f"file not found: {path}")
        lines = p.read_text(errors="replace").splitlines()
        lo, hi = 1, len(lines)
        if rng and len(rng) == 2:
            lo, hi = max(1, int(rng[0])), min(len(lines), int(rng[1]))
        numbered = "\n".join(f"{i:>4}: {lines[i-1]}" for i in range(lo, hi + 1))
        # A real repo has files a micro-task never did: sqlglot's `parser.py`
        # is 415 KB (~104k tokens), i.e. three times the model's whole window
        # from a single `view`. Uncapped, that history was silently truncated
        # from the left at serving time (`KVEngine`'s `p[-max_len:]`), so the
        # model answered the tail of a source file with no system prompt and
        # no issue — and the pair built from it taught that. Cap at the same
        # budget as every other tool result and point at `view_range`.
        if len(numbered) > MAX_OUTPUT:
            kept, n = [], 0
            for line in numbered.splitlines():
                if n + len(line) + 1 > MAX_OUTPUT:
                    break
                kept.append(line)
                n += len(line) + 1
            numbered = "\n".join(kept) + (
                f"\n... [view truncated at {MAX_OUTPUT} chars; this file has "
                f"{len(lines)} lines — use view_range [start, end] for more]")
        return {"path": str(p), "content": numbered, "total_lines": len(lines)}

    def create(self, path: str, file_text: str) -> dict:
        from agent_env import ToolError
        if file_text is None:
            raise ToolError("create: `file_text` is required (send \"\" for an empty file)")
        p = self.resolve(path)
        if p.exists():
            raise ToolError(f"file already exists: {path}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text)
        return {"path": str(p), "created": True}

    def str_replace(self, path: str, old_str: str, new_str: str) -> dict:
        from agent_env import ToolError
        if not old_str:
            raise ToolError("str_replace: `old_str` is required and must be non-empty")
        if not new_str:
            raise ToolError("str_replace: `new_str` is required and must be non-empty "
                            "(this call would delete the matched text)")
        p = self.resolve(path)
        if not p.exists():
            raise ToolError(f"file not found: {path}")
        text = p.read_text()
        if text.count(old_str) != 1:
            raise ToolError(f"old_str must occur exactly once; found {text.count(old_str)}")
        p.write_text(text.replace(old_str, new_str))
        return {"path": str(p), "replaced": True}

    def insert(self, path: str, insert_line: int, new_str: str) -> dict:
        from agent_env import ToolError
        if not new_str:
            raise ToolError("insert: `new_str` is required and must be non-empty")
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
        if self.ephemeral and not self.keep:
            shutil.rmtree(self.workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------
_JEST_JSON = ".jest-out.json"
_PYTEST_XML = ".pytest-out.xml"


def parse_jest(text: str) -> dict:
    """`--json` report -> {passed, total, failed:[names], ran:[names]}.

    Only used when the runner actually wrote a report; the exit code is still
    the authority when it did not (a collection error never writes one).
    """
    try:
        d = json.loads(text)
    except Exception:  # noqa: BLE001
        return {}
    ran, failed = [], []
    for tr in d.get("testResults", []):
        for a in tr.get("assertionResults", []):
            name = " ".join(x for x in (a.get("ancestorTitles") or []) + [a.get("title")] if x)
            ran.append(name)
            if a.get("status") != "passed":
                failed.append(name)
    return {"passed": bool(d.get("success")) and not failed, "total": len(ran),
            "failed": failed, "ran": ran}


def parse_pytest(text: str) -> dict:
    """pytest's `--junitxml` -> the same shape as `parse_jest`.

    A collection error is reported by pytest as a `<testcase>` carrying an
    `<error>`, so it lands in `failed` and `total` stays truthful — which is the
    difference between "the fix's test fails" (gradable) and "the module does not
    import" (an environment bug that must not pass for a real failure).
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    ran, failed = [], []
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        cls = tc.get("classname") or ""
        full = f"{cls}::{name}" if cls else name
        ran.append(full)
        if tc.find("failure") is not None or tc.find("error") is not None:
            failed.append(full)
    if not ran:
        return {}
    return {"passed": not failed, "total": len(ran), "failed": failed, "ran": ran}


def parse_test_result(inst: RepoInstance, res: dict, repo_dir) -> dict:
    """The SWE-bench rule: the fix commit's tests must pass.

    `check_cmd` is written so that a machine-readable report is the primary
    evidence and the exit code is the fallback, because "the runner exited 0" is
    spoofable by a model that deletes the test file (which
    `install_hidden_tests` has already undone) and because a per-test view is
    what the pairs need. The pytest path exists because exit-code-only gating
    cannot tell "the fix's test failed" from "the suite did not collect": the
    first probe returned `n_tests=0->0` for every Python candidate, which would
    have let a broken environment pass as a verified instance.
    """
    py = getattr(inst, "lang", "") == "py"
    out_file = Path(repo_dir) / (_PYTEST_XML if py else _JEST_JSON)
    report = {}
    if out_file.exists():
        try:
            text = out_file.read_text()
            report = parse_pytest(text) if py else parse_jest(text)
        except (OSError, ET.ParseError):
            report = {}
    if report:
        passed = report["passed"]
    else:
        passed = res.get("exit_code") == 0
    return {"ran": True, "passed": bool(passed), "cmd": inst.check_cmd,
            "exit_code": res.get("exit_code"),
            "failed_tests": report.get("failed", []),
            "n_tests": report.get("total", 0),
            "output": ((res.get("stdout") or "") + (res.get("stderr") or ""))[-1500:]}


def repo_check(task, sb) -> dict:
    """`run_check` for real repos: hidden tests in, then run.

    Installed fresh on every call, so this is valid both as the episode's
    pristine `initial_check` (base state → the hidden tests fail) and as the
    final check (model's edits → pass iff it solved it).

    The changed-path snapshot is taken *before* the hidden tests land, because
    overwriting a test file the model edited makes `git diff` attribute the
    hidden version to the model — the grade would then report a test edit that
    did not happen.
    """
    sb.pre_check_changed = sb.changed()
    install_hidden_tests(task.inst, sb.workdir)
    res = sb.bash(task.check_cmd, timeout=TEST_TIMEOUT)
    return parse_test_result(task.inst, res, sb.workdir)


SHORTCUT_RE = re.compile(
    r"\b(if\s+.*\bexpect\(|process\.env\.CI\s*=|--forceExit|test\.skip|"
    r"describe\.skip|it\.skip|xit\(|xdescribe\()", re.I)


def grade_repo(task, sb, episode) -> dict:
    """Score one finished real-repo episode. Sandbox must still be open.

    Analysis only — pass/fail is already in `episode["check"]`, which ran on the
    restored-test tree. This adds the two things DPO needs beside the label:
    whether the model touched test files (it must not profit from it) and
    whether it left the source in a state that does not even parse, because
    those are the failure modes round 5 paid for.
    """
    inst = task.inst
    changed = sb.diff_initial()
    tests = set(inst.test_files)
    touched = getattr(sb, "pre_check_changed", None)
    if touched is None:
        touched = {r: True for r in changed}
    tests_changed = sorted(r for r in touched if r in tests)
    source_changed = sorted(r for r in changed if r not in tests)
    shortcuts = []
    for rel in source_changed:
        _b, after = changed[rel]
        for line in (after or "").splitlines():
            if SHORTCUT_RE.search(line):
                shortcuts.append(f"{rel}: {line.strip()[:80]}")
    check = (episode or {}).get("check") or {}
    return {
        "passed": bool(check.get("passed")),
        "failed_tests": check.get("failed_tests", []),
        "tests_changed": tests_changed,
        "source_changed": source_changed,
        "shortcuts": shortcuts,
        "edited_tests_only": bool(tests_changed) and not source_changed,
        "gold_files_touched": sorted(set(source_changed) & set(inst.source_files)),
        "check": check,
    }


# ---------------------------------------------------------------------------
# task + episode
# ---------------------------------------------------------------------------
class RepoTask:
    """Duck-types `agent_env.MicroTask` for `run_episode`/`execute_tool`.

    `files` is deliberately empty: the tree already exists, and anything that
    *needs* pristine test text asks the fix commit for it (see
    `install_hidden_tests`). Carrying a dict of the repo would defeat the point.
    """

    def __init__(self, inst: RepoInstance, tier=0):
        self.inst = inst
        self.task_id = inst.instance_id
        self.files: dict = {}
        self.problem = inst.problem
        self.check_cmd = inst.check_cmd
        self.fix = None
        self.tier = tier
        self.test_files = list(inst.test_files)
        self.variant = None

    def user_prompt(self, workdir: str) -> str:
        return f"{self.problem}\n\nThe repository is at {workdir}."


def run_repo_episode(inst: RepoInstance, policy, root=None, max_steps=None,
                     keep=True, grader=None, path_prefixes=()) -> dict:
    """One full episode in the real repo, on a fresh checkout of the base commit.

    The tree is reset *before* the policy moves, so episodes never inherit each
    other's edits; `check_cmd`'s hidden tests are installed by the check itself.
    """
    from agent_env import MAX_STEPS, run_episode

    repo_dir = prepare(inst, root=root)
    reset_tree(repo_dir)
    sb = RepoSandbox(repo_dir, keep=True, path_prefixes=path_prefixes)
    try:
        ep = run_episode(RepoTask(inst), policy, max_steps=max_steps or MAX_STEPS,
                         keep=keep, grader=grader or grade_repo, sandbox=sb,
                         check_fn=repo_check)
    finally:
        # the tree outlives the episode: node_modules is the expensive part
        sb.keep = True
    ep["instance_id"] = inst.instance_id
    ep["workdir"] = str(repo_dir)
    return ep


def selftest() -> int:
    """Pure-logic checks: report parsing and change accounting, no model."""
    bad = 0
    rep = parse_jest(json.dumps({"success": False, "testResults": [
        {"assertionResults": [
            {"title": "a", "ancestorTitles": ["suite"], "status": "passed"},
            {"title": "b", "ancestorTitles": [], "status": "failed"}]}]}))
    if rep != {"passed": False, "total": 2, "failed": ["b"], "ran": ["suite a", "b"]}:
        print(f"  FAIL parse_jest: {rep}")
        bad += 1

    rep = parse_jest(json.dumps({"success": True, "testResults": [
        {"assertionResults": [{"title": "a", "ancestorTitles": [], "status": "passed"}]}]}))
    if not rep["passed"] or rep["total"] != 1:
        print(f"  FAIL parse_jest pass case: {rep}")
        bad += 1

    if parse_jest("not json") != {}:
        print("  FAIL parse_jest should ignore a missing report")
        bad += 1

    # pytest's junit report: a failing test, a passing test, and a collection
    # error (which must land in `failed`, not vanish into an exit code)
    rep = parse_pytest(
        '<testsuites><testsuite name="s">'
        '<testcase classname="tests.test_a" name="test_ok"/>'
        '<testcase classname="tests.test_a" name="test_bad"><failure/></testcase>'
        '<testcase classname="tests.test_b" name="tests.test_b"><error/></testcase>'
        '</testsuite></testsuites>')
    if rep.get("total") != 3 or rep.get("passed") is not False:
        print(f"  FAIL parse_pytest: {rep}")
        bad += 1
    if rep.get("failed") != ["tests.test_a::test_bad", "tests.test_b::tests.test_b"]:
        print(f"  FAIL parse_pytest failed names: {rep.get('failed')}")
        bad += 1
    rep = parse_pytest('<testsuites><testsuite><testcase classname="a" '
                       'name="test_ok"/></testsuite></testsuites>')
    if not rep.get("passed") or rep.get("total") != 1:
        print(f"  FAIL parse_pytest pass case: {rep}")
        bad += 1
    if parse_pytest("<not xml") != {} or parse_pytest("<testsuites/>") != {}:
        print("  FAIL parse_pytest should ignore an empty/unreadable report")
        bad += 1

    # a model that only edits tests is not a solver
    inst = RepoInstance("x", "r", "b", "f", "p", ["t.js"], ["s.js"], "true")
    task = RepoTask(inst)
    class _SB:
        def diff_initial(self):
            return {"t.js": ("old", "new")}
    g = grade_repo(task, _SB(), {"check": {"passed": False}})
    if not g["edited_tests_only"] or g["passed"]:
        print(f"  FAIL edited_tests_only: {g}")
        bad += 1

    # the install marker must sit outside the tree: `reset_tree` runs
    # `git clean -fd`, which deleted it and made every episode reinstall
    if _marker(inst, Path("/tmp/root/inst-x")).parent != Path("/tmp/root"):
        print(f"  FAIL marker lives in the tree: {_marker(inst, Path('/tmp/root/inst-x'))}")
        bad += 1

    # a real repo's file can dwarf the context (sqlglot/parser.py: 415 KB), so
    # `view` caps like every other tool result and says how to page
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "r"
        repo.mkdir()
        for cmd in (["git", "init", "-q"],
                    ["git", "config", "user.email", "a@b"],
                    ["git", "config", "user.name", "a"]):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
        (repo / "big.py").write_text(
            "\n".join(f"line {i} " + "x" * 40 for i in range(2000)))
        for cmd in (["git", "add", "-A"], ["git", "commit", "-qm", "x"]):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
        sb = RepoSandbox(repo)
        v = sb.view("big.py")
        if len(v["content"]) > MAX_OUTPUT + 200 or "view_range" not in v["content"]:
            print(f"  FAIL view cap: {len(v['content'])} chars")
            bad += 1
        if v["total_lines"] != 2000:
            print(f"  FAIL view total_lines: {v['total_lines']}")
            bad += 1
        v2 = sb.view("big.py", [1990, 2000])
        if "view truncated" in v2["content"] or "line 1999" not in v2["content"]:
            print(f"  FAIL view_range: {v2['content'][-80:]!r}")
            bad += 1

    if not _TEST_FILES_RE.search("src/action.test.js") or _TEST_FILES_RE.search("src/action.js"):
        print("  FAIL test-file regex")
        bad += 1

    print("selftest:", "ok" if not bad else f"{bad} failure(s)")
    return 1 if bad else 0


# test-file detection, shared with the instance builder (one rule, one place)
_TEST_FILES_RE = re.compile(
    r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*$|[._-]test\.|\.test\.|"
    r"[._-]spec\.|Test\.java$", re.I)

# Extensions a harness can actually run. Kept here because both the env and the
# instance builder need the same answer to "is this file a test".
SOURCE_EXT = {".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx", ".py", ".go", ".rb",
              ".php", ".java", ".rs"}


def is_test_file(path: str) -> bool:
    """A *runnable* test: test-shaped path **and** a source extension.

    The extension half matters. `.github/workflows/local-test.yaml` is a path
    with "test" in it and no harness can run it; counting it as a test made
    "Upgrade Node.js to 24" look like a unit-test instance whose only test was
    a CI workflow file.
    """
    return (bool(_TEST_FILES_RE.search(path))
            and Path(path).suffix.lower() in SOURCE_EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--instances", help="jsonl from rl/realinst.py")
    ap.add_argument("--root", default=None)
    ap.add_argument("--prepare", nargs="*", help="instance ids to clone+install")
    ap.add_argument("--check", nargs="*", help="instance ids to grade as-is (expect fail)")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.instances:
        ap.print_help()
        return 2

    insts = {i.instance_id: i for i in load_instances(a.instances)}
    print(f"{len(insts)} instances from {a.instances}")
    for iid in (a.prepare or []):
        d = prepare(insts[iid], root=a.root)
        print(f"  prepared {iid} -> {d}")
    for iid in (a.check or []):
        inst = insts[iid]
        repo_dir = prepare(inst, root=a.root)
        sb = RepoSandbox(repo_dir, keep=True)
        got = repo_check(RepoTask(inst), sb)
        print(f"  {iid}: passed={got['passed']} n_tests={got.get('n_tests')} "
              f"failed={got.get('failed_tests')[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
