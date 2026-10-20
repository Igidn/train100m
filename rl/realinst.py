#!/usr/bin/env python3
"""Build real-repo instances from a repo's own fix commits — MiMo's construction.

Verified on `format-code-task-001457` (2026-10-20): the MiMo image is a
445 MB single-layer rootfs holding `hashicorp/vault-action` at `6728115` with
`node_modules` pre-installed and a full `.git`. The row's prompt is the GitHub
issue, and that commit is an ancestor of the upstream fix:

    c14a190  feat(): make secrets parameter optional (#299)
      src/action.js      | 6 +++++-     <- gold patch
      src/action.test.js | 9 +++++++++  <- test patch
    git apply --check <test patch> at 6728115  ->  applies clean

So an instance is not a dataset row to be found, it is a *fix commit to be
inverted*: base = the fix's parent, gold = the fix's non-test files, hidden
tests = the fix's test files, and the prompt = the issue the fix closed. That is
also how the verifier stays honest — every instance is gated on the fix itself
(base + hidden tests must FAIL, + gold must PASS) before it is allowed into the
pair set, so no instance can be ungradable by construction.

    python rl/realinst.py --repo https://github.com/hashicorp/vault-action \\
        --out rl/instances/vault-action.jsonl --limit 10 --gate
    python rl/realinst.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from realenv import (SOURCE_EXT, RepoInstance, RepoSandbox, RepoTask,  # noqa: E402
                     apply_gold, is_test_file, prepare, repo_check, reset_tree,
                     _git, _run)

# ---------------------------------------------------------------------------
# commit selection
# ---------------------------------------------------------------------------
SKIP_RE = re.compile(
    r"revert|\[compliance\]|chore\(deps\)|bump |dependabot|^merge |changelog|"
    r"^update to v?[0-9]|^release |^v?[0-9]+\.[0-9]+\.[0-9]+$", re.I)
# files that are never the point of a task: docs, CI, lockfiles, config
NON_CODE_RE = re.compile(
    r"^(docs?|examples?|dist|build|vendor|\.github)/|\.(md|txt|json|lock|yml|yaml|rst|cfg|toml)$|"
    r"(^|/)(README|CHANGELOG|LICENSE|NOTICE|package-lock|yarn\.lock|poetry\.lock|"
    r"Makefile|Dockerfile|setup\.cfg|pyproject\.toml|\.gitignore|\.eslintrc.*)$", re.I)
SRC_EXT = SOURCE_EXT     # one definition, in the env
# Tests that need a live service (docker-compose, a Vault server, a browser) are
# not gradable in a bare worktree. MiMo's images are; ours are not, so a
# candidate whose only tests are integration tests is dropped rather than
# gated into a permanent failure — `format-code-task-001457`'s own row is
# `src/action.test.js` (a unit test), while `make "role" input optional` only
# touches `integrationTests/basic/jwt_auth.test.js`, which wants a real Vault.
INTEGRATION_RE = re.compile(
    r"(^|/)(integrationTests?|e2e|e2e-tls|functional|acceptance|smoke)/|"
    r"(^|/)(integration|e2e)[._-]|\.(integration|e2e)\.", re.I)
TEST_NAME_RES = {
    "js": [re.compile(r"""(?:test|it)\(\s*["'`](.+?)["'`]"""),
           re.compile(r"""describe\(\s*["'`](.+?)["'`]""")],
    "py": [re.compile(r"""^\+\s*def\s+(test_\w+)""")],
    "go": [re.compile(r"""^\+\s*func\s+(Test\w+)""")],
}


def is_code_file(path: str) -> bool:
    if NON_CODE_RE.search(path):
        return False
    return Path(path).suffix in SRC_EXT


def pr_number(subject: str):
    """`... (#299)` -> 299. Squash merges carry it; nothing else does."""
    m = re.search(r"\(#(\d+)\)\s*$", subject.strip())
    return int(m.group(1)) if m else None


def split_files(files: list) -> tuple:
    return ([f for f in files if is_code_file(f) and not is_test_file(f)],
            [f for f in files if is_test_file(f)])


def extract_fail_to_pass(patch: str, lang: str) -> list:
    """Test names the patch *adds* — SWE-bench's FAIL_TO_PASS, by convention."""
    names = []
    for line in patch.splitlines():
        if not line.startswith("+"):
            continue
        for rx in TEST_NAME_RES.get(lang, []):
            m = rx.search(line)
            if m:
                names.append(m.group(1).strip())
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# language plumbing
# ---------------------------------------------------------------------------
def detect_lang(repo_dir) -> str:
    if (Path(repo_dir) / "package.json").exists():
        return "js"
    if list(Path(repo_dir).glob("*.py")) or (Path(repo_dir) / "setup.py").exists():
        return "py"
    if (Path(repo_dir) / "go.mod").exists():
        return "go"
    return "js"


def test_command(lang: str, test_files: list) -> str:
    """One rule: JSON report when the runner has one, exit code otherwise."""
    files = " ".join(f"'{f}'" for f in test_files)
    if lang == "js":
        return (f"npx jest --ci --silent --json "
                f"--outputFile=.jest-out.json {files}")
    if lang == "py":
        return f"python -m pytest -q {files}"
    if lang == "go":
        return f"go test {' '.join(f'./{Path(f).parent}' for f in test_files)}"
    return f"echo 'no test runner for lang={lang}' && false"


def install_command(lang: str, repo_dir) -> str | None:
    if lang == "js":
        if (Path(repo_dir) / "package-lock.json").exists():
            return "npm ci"
        return "npm install"
    if lang == "py":
        return "python -m pip install -e . -q"
    return None


# ---------------------------------------------------------------------------
# source clone + candidates
# ---------------------------------------------------------------------------
def repo_slug(repo: str) -> str | None:
    m = re.search(r"github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?/?$", repo.strip())
    return f"{m.group(1)}/{m.group(2)}" if m else None


def ensure_source_clone(repo: str, root) -> Path:
    """A full clone, kept under `<root>/src/<name>` so a rerun is free."""
    name = (repo_slug(repo) or Path(repo).name).replace("/", "__")
    dest = Path(root) / "src" / name
    if dest.exists():
        _git(dest, "fetch", "--quiet", "--all", check=False)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--quiet", repo, str(dest)], check=True, timeout=1800)
    return dest


def candidates(src_dir, window=None, branch=None, unit_only=True,
               max_gold=None) -> list:
    """Fix commits worth inverting: code + tests, not reverts or dep bumps.

    `unit_only` keeps the tests that can run in a bare worktree (see
    `INTEGRATION_RE`); a candidate with no such test is not gradable and is
    skipped before any install is paid for.
    """
    ref = branch or "HEAD"
    out = _git(src_dir, "log", "--first-parent", "--no-merges",
               "--format=%H\t%ci\t%s", ref)
    rows = []
    for line in out.splitlines():
        if not line.strip():
            continue
        sha, date, subject = line.split("\t", 2)
        day = date[:10]
        if window and not (window[0] <= day <= window[1]):
            continue
        if SKIP_RE.search(subject):
            continue
        files = [f for f in _git(src_dir, "show", "--name-only", "--format=", sha).splitlines() if f.strip()]
        source, tests = split_files(files)
        if unit_only:
            tests = [t for t in tests if not INTEGRATION_RE.search(t)]
        if not source or not tests:
            continue
        patch = _git(src_dir, "diff", f"{sha}^", sha, "--", *source)
        gold_lines = sum(1 for l in patch.splitlines()
                         if l[:1] in "+-" and l[:3] not in ("+++", "---"))
        if max_gold is not None and gold_lines > max_gold:
            continue
        rows.append({"sha": sha, "date": date, "subject": subject,
                     "source_files": source, "test_files": tests,
                     "gold_lines": gold_lines})
    return rows


def make_instance(src_dir, cand, repo, lang=None) -> RepoInstance:
    sha = cand["sha"]
    lang = lang or detect_lang(src_dir)
    test_patch = _git(src_dir, "diff", f"{sha}^", sha, "--", *cand["test_files"])
    inst = RepoInstance(
        instance_id=f"{repo_slug(repo) or 'local'}-{sha[:8]}".replace("/", "-"),
        repo=repo,
        base_commit=_git(src_dir, "rev-parse", f"{sha}^").strip(),
        fix_commit=_git(src_dir, "rev-parse", sha).strip(),
        problem="",                                  # filled after the gate
        test_files=cand["test_files"],
        source_files=cand["source_files"],
        check_cmd=test_command(lang, cand["test_files"]),
        install_cmd=install_command(lang, src_dir),
        lang=lang,
        fail_to_pass=extract_fail_to_pass(test_patch, lang),
        meta={"subject": cand["subject"], "date": cand["date"],
              "gold_lines": cand["gold_lines"]},
    )
    return inst


# ---------------------------------------------------------------------------
# the problem statement: the issue the fix closed
# ---------------------------------------------------------------------------
def _gh(path: str, token=None):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={"Accept": "application/vnd.github+json",
                 "User-Agent": "tinyballs-realinst/1",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


CLOSE_RE = re.compile(r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s*:?\s*#(\d+)", re.I)


def problem_from_pr(slug: str, pr: int, token=None) -> str:
    """The issue text where we can find it, the PR body otherwise.

    MiMo's prompt for `format-code-task-001457` is the *issue*, not the PR
    ("[FEAT] Make `secrets` parameter optional..."), so a PR body that says
    "fixes #NNN" is followed once before falling back.
    """
    if not slug or not pr:
        return ""
    try:
        pull = _gh(f"/repos/{slug}/pulls/{pr}", token)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return ""
    body = pull.get("body") or ""
    m = CLOSE_RE.search(body)
    if m:
        try:
            issue = _gh(f"/repos/{slug}/issues/{m.group(1)}", token)
            title, ibody = issue.get("title") or "", issue.get("body") or ""
            if title or ibody:
                return f"{title}\n\n{ibody}".strip()
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
            pass
    title = pull.get("title") or ""
    if body:
        return f"{title}\n\n{body}".strip()
    return title


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------
def gate_instance(inst: RepoInstance, root) -> dict:
    """base+hidden tests must fail, gold+hidden tests must pass.

    Without this the pair set can contain a task that was already solved at the
    base commit (a free pass) or one the fix does not actually fix (unwinnable by
    anyone, so its "rejected" side says nothing).
    """
    repo_dir = prepare(inst, root=root)
    sb = RepoSandbox(repo_dir, keep=True)
    reset_tree(repo_dir)
    before = repo_check(RepoTask(inst), sb)      # hidden tests, base source -> must fail
    reset_tree(repo_dir)                         # drop the hidden tests before gold
    apply_gold(inst, repo_dir)                   # the reference solution
    after = repo_check(RepoTask(inst), sb)       # hidden tests again -> must pass
    reset_tree(repo_dir)
    return {
        "fail_before": not before["passed"],
        "pass_after": bool(after["passed"]),
        "failed_before": before.get("failed_tests", [])[:8],
        "n_tests_before": before.get("n_tests", 0),
        "n_tests_after": after.get("n_tests", 0),
        "fail_to_pass_hit": [t for t in inst.fail_to_pass
                             if any(t in f for f in before.get("failed_tests", []))],
        "output_after": after.get("output", "")[-400:],
    }


# ---------------------------------------------------------------------------
def build(args) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    src = ensure_source_clone(args.repo, root)
    slug = repo_slug(args.repo)
    window = tuple(args.window.split(":")) if args.window else None

    cands = candidates(src, window=window, branch=args.branch,
                       unit_only=not args.allow_integration,
                       max_gold=args.max_gold)
    if args.sort == "small":
        cands.sort(key=lambda c: c["gold_lines"])
    elif args.sort == "old":
        cands.reverse()
    print(f"{args.repo}: {len(cands)} candidate fix commits"
          f"{f' in {window[0]}..{window[1]}' if window else ''}")
    for c in cands[:12]:
        print(f"  {c['sha'][:8]} {c['date'][:10]} gold={c['gold_lines']:>3}L "
              f"tests={len(c['test_files'])} {c['subject'][:58]}")

    token = os.environ.get("GITHUB_TOKEN")
    kept, tried = [], 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = root / "problems.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    for cand in cands:
        if len(kept) >= args.limit or tried >= args.max_tries:
            break
        tried += 1
        inst = make_instance(src, cand, args.repo)
        if args.gate:
            try:
                g = gate_instance(inst, root)
            except Exception as e:                       # noqa: BLE001
                print(f"  {inst.instance_id}: gate error {type(e).__name__}: {e}")
                continue
            ok = g["fail_before"] and g["pass_after"]
            print(f"  {inst.instance_id}: fail_before={g['fail_before']} "
                  f"pass_after={g['pass_after']} n_tests={g['n_tests_before']}->"
                  f"{g['n_tests_after']} f2p_hit={len(g['fail_to_pass_hit'])}"
                  f"/{len(inst.fail_to_pass)} {'KEEP' if ok else 'drop'}")
            inst.meta["gate"] = g
            if not ok:
                continue
        key = inst.instance_id
        if key not in cache:
            cache[key] = problem_from_pr(slug, pr_number(cand["subject"]), token)
            cache_path.write_text(json.dumps(cache, indent=1))
        inst.problem = cache[key] or cand["subject"]
        kept.append(inst)

    with open(out_path, "w") as fh:
        for inst in kept:
            fh.write(json.dumps(inst.to_json()) + "\n")
    print(f"\nwrote {len(kept)} instances -> {out_path} ({tried} candidates tried)")
    for inst in kept:
        print(f"  {inst.instance_id} tests={inst.test_files} f2p={len(inst.fail_to_pass)}")
    return 0 if kept else 1


def selftest() -> int:
    bad = 0
    if pr_number("feat(): make secrets parameter optional (#299)") != 299:
        print("  FAIL pr_number"); bad += 1
    if pr_number("no number here") is not None:
        print("  FAIL pr_number none"); bad += 1
    if SKIP_RE.search("Revert \"fix secrets stored in json format (#466)\" (#471)") is None:
        print("  FAIL skip revert"); bad += 1
    if SKIP_RE.search("fix secrets stored in JSON format (#473)"):
        print("  FAIL skip too greedy"); bad += 1
    src, tests = split_files(["src/action.js", "src/action.test.js", "README.md",
                              "package-lock.json", "action.yml"])
    if src != ["src/action.js"] or tests != ["src/action.test.js"]:
        print(f"  FAIL split_files: {src} {tests}"); bad += 1
    if not is_test_file("tests/test_foo.py") or is_test_file("src/foo.py"):
        print("  FAIL is_test_file"); bad += 1
    names = extract_fail_to_pass(
        "+  test('exports token when secrets omitted', () => {\n"
        "+    it(\"keeps working\", () => {})\n"
        "-  test('old', () => {})\n", "js")
    if names != ["exports token when secrets omitted", "keeps working"]:
        print(f"  FAIL extract_fail_to_pass: {names}"); bad += 1
    if problem_from_pr("", None) != "":
        print("  FAIL problem_from_pr no slug"); bad += 1
    print("selftest:", "ok" if not bad else f"{bad} failure(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="clone url or local path")
    ap.add_argument("--out", default="rl/instances/instances.jsonl")
    ap.add_argument("--root", default=os.environ.get("REALENV_ROOT",
                                                     "/tmp/realinst"))
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--window", default=None, help="YYYY-MM-DD:YYYY-MM-DD")
    ap.add_argument("--branch", default=None)
    ap.add_argument("--sort", choices=["small", "new", "old"], default="small")
    ap.add_argument("--gate", action="store_true",
                    help="run the fail-before/pass-after gate (slow, required)")
    ap.add_argument("--allow-integration", action="store_true",
                    help="keep candidates whose tests need a live service")
    ap.add_argument("--max-gold", type=int, default=200,
                    help="skip fixes whose gold patch is longer than this "
                         "(a 26k-line dependency rewrite is not a task)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.repo:
        ap.print_help()
        return 2
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main())
