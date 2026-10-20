#!/usr/bin/env python3
"""Real-repo DPO pairs, both sides verified by the repo's own tests.

The shape follows Stage A (`rl/build_pairs.py`) — turn-level, teacher-forced
pairs: at a state the *gold* trajectory actually visited, chosen is the gold
action and rejected is what our model does at that same prefix. What changes is
the ground truth: the micro-sandbox is replaced by a real checkout running the
repo's real test suite (`rl/realenv.py`), and both sides are graded by it.

    chosen   = a scripted replay of the upstream fix, executed for real
               (`git diff <fix>^ <fix>` -> str_replace calls -> the suite runs)
    rejected = our model's own turn at that prefix
    instance = dropped entirely if our model solved it

That last rule is the "do not remove behaviour that is already there" gate, and
it is deliberately blunt: a model that passes the hidden suite may have passed
by a *different* correct route, and pairing its turns against the gold fix would
teach it out of a working behaviour. So a solved instance contributes nothing,
not a rejected side. (The alternative — keeping its passing episode as a chosen
side — needs no pair at all; it is already in the SFT/positive pool.)

No teacher is involved. The chosen side is the human fix, which is the strongest
"good data" available for these instances and costs no GPU: the MiMo code rows
are upstream-PR instances (base = the fix's parent, tests = the fix's tests), and
`rl/realinst.py` has already gated every instance on
`base+hidden tests -> FAIL, + gold -> PASS` before it reaches this file.

    # local, against the Stage-A/DPO checkpoint on Ollama
    python rl/gen_real_pairs.py --student tinyballs-dpo \\
        --instances rl/instances/vault-action.jsonl \\
        --out rl/samples/real-pairs.jsonl --episodes rl/samples/real-episodes.jsonl

    # no student: only prove the gold replay solves each instance
    python rl/gen_real_pairs.py --gold-only --instances ...

    # test the diff -> tool-call translation without touching a repo
    python rl/gen_real_pairs.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_env import MAX_STEPS, TOOL_CALL  # noqa: E402
from agent_rollout import OllamaPolicy, _steps_policy  # noqa: E402
from build_pairs import make_query, pairs_from_trajectory  # noqa: E402
from realenv import (_git, load_instances, prepare, run_repo_episode)  # noqa: E402


# ---------------------------------------------------------------------------
# gold patch -> the tool calls that reproduce it
# ---------------------------------------------------------------------------
def _call(name: str, **args) -> str:
    return TOOL_CALL + json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def _str_replace(path: str, old: str, new: str) -> str:
    return _call("str_replace_editor", command="str_replace", path=path,
                 old_str=old, new_str=new)


def parse_patch(patch: str) -> list:
    """unified diff -> [{path, new_file, deleted, hunks:[{old:[],new:[]}]}].

    Only the shape `git diff` emits for text files is handled, and only the
    first char of a line is trusted inside a hunk — a source line may itself
    start with `+++` and must not be read as a header (a state machine, not
    `startswith("--- ")` on every line).
    """
    files, cur, hunk = [], None, None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            cur = {"path": None, "new_file": False, "deleted": False, "hunks": []}
            files.append(cur)
            hunk = None
            continue
        if cur is None:
            continue
        if hunk is None:
            if line.startswith("new file mode"):
                cur["new_file"] = True
            elif line.startswith("deleted file mode"):
                cur["deleted"] = True
            elif line.startswith("+++ "):
                p = line[4:].strip()
                cur["path"] = p[2:] if p.startswith("b/") else p
            elif line.startswith("@@"):
                hunk = {"old": [], "new": []}
                cur["hunks"].append(hunk)
            continue
        if line.startswith("@@"):        # a second hunk in the same file
            hunk = {"old": [], "new": []}
            cur["hunks"].append(hunk)
            continue
        if line.startswith("+"):
            hunk["new"].append(line[1:])
        elif line.startswith("-"):
            hunk["old"].append(line[1:])
        elif line.startswith(" "):
            hunk["old"].append(line[1:])
            hunk["new"].append(line[1:])
        # "\ No newline at end of file" carries no text; both sides end there
    return [f for f in files if f["path"]]


def _block(lines) -> str:
    return "\n".join(lines) + "\n" if lines else ""


def edit_steps(inst, repo_dir, show=None) -> list:
    """The gold patch as `str_replace` calls, hunk by hunk.

    A hunk is only used when its old text occurs exactly once in the base file
    (the editor's own rule), otherwise that file falls back to a single
    whole-file replace — which is always unique, and still a real edit. Nothing
    here guesses: every candidate string was checked against the base blob.
    """
    diff = _git(repo_dir, "diff", f"{inst.fix_commit}^", inst.fix_commit,
                "--", *inst.source_files)
    steps = []
    for f in parse_patch(diff):
        path = f["path"]
        if f["deleted"]:
            continue
        base = show(path) if show else _git(repo_dir, "show",
                                            f"{inst.base_commit}:{path}", check=False)
        fixed = _git(repo_dir, "show", f"{inst.fix_commit}:{path}", check=False)
        if f["new_file"] or not f["hunks"]:
            steps.append(_call("str_replace_editor", command="create",
                               path=path, file_text=fixed))
            continue
        hunked = []
        for h in f["hunks"]:
            old, new = _block(h["old"]), _block(h["new"])
            hunks_text = "\n".join(h["old"])
            if not old or base.count(old) != 1 or (hunks_text and base.count(hunks_text) != 1):
                hunked = []
                break
            hunked.append(_str_replace(path, old, new))
        steps.extend(hunked or [_str_replace(path, base, fixed)])
    return steps


def gold_steps(inst, repo_dir, max_steps=MAX_STEPS) -> list | None:
    """Inspect -> edit -> run the suite -> submit, within the step budget.

    Views are what get dropped when the budget is tight: an edit that does not
    fit means the fix cannot be expressed, so that instance is refused rather
    than replayed half-way.
    """
    edits = edit_steps(inst, repo_dir)
    if not edits:
        return None
    head = [_call("bash", command="ls")]
    views = [_call("str_replace_editor", command="view", path=p)
             for p in (list(inst.test_files[:1]) + list(inst.source_files[:2]))]
    tail = [_call("bash", command=inst.check_cmd), _call("submit")]
    while views and len(head) + len(views) + len(edits) + len(tail) > max_steps:
        views.pop()
    while head and len(head) + len(views) + len(edits) + len(tail) > max_steps:
        head.pop()
    if len(head) + len(views) + len(edits) + len(tail) > max_steps:
        return None
    return head + views + edits + tail


def gold_policy(inst, repo_dir):
    steps = gold_steps(inst, repo_dir)
    if steps is None:
        return None, []
    return _steps_policy([{"text": s, "stopped": True, "calls": []} for s in steps]), steps


# ---------------------------------------------------------------------------
# pair kind (so the trainer can filter reads out of a first pass)
# ---------------------------------------------------------------------------
def call_kind(chosen_text: str) -> str:
    import re
    m = re.search(r'"name"\s*:\s*"([^"]+)"', chosen_text)
    if not m:
        return "none"
    name = m.group(1)
    if name == "submit":
        return "submit"
    if name == "bash":
        return "run"
    if name == "str_replace_editor":
        return "edit" if re.search(r'"command"\s*:\s*"(create|str_replace|insert)"', chosen_text) else "read"
    return name


# ---------------------------------------------------------------------------
# context budget
# ---------------------------------------------------------------------------
def _chars(ep) -> int:
    return sum(len(m.get("content") or "") for m in ep.get("messages") or [])


def _over_budget(ep, max_chars: int) -> bool:
    return _chars(ep) > max_chars


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def run(args) -> int:
    insts = load_instances(args.instances)
    if args.limit:
        insts = insts[:args.limit]
    if args.only:
        insts = [i for i in insts if any(o in i.instance_id for o in args.only)]
    print(f"{len(insts)} instances from {args.instances}")

    tok = None
    if args.tokenizer:
        from tokenizers import Tokenizer
        path = args.tokenizer
        if os.path.isdir(path):            # an HF export dir carries tokenizer.json
            path = os.path.join(path, "tokenizer.json")
        tok = Tokenizer.from_file(path)
        print(f"tokenizer: {path}")

    policy = None
    if not args.gold_only:
        if args.student.startswith("hf:"):
            from agent_rollout import HFPolicy
            policy = HFPolicy(model_dir=args.student[3:], temp=args.temp,
                              max_len=args.num_ctx, seed=args.seed,
                              max_tokens=args.max_tokens)
        else:
            # Accept the `ollama:` prefix that every other entry point uses, so
            # `--student ollama:tinyballs-grpo6` is not sent to Ollama verbatim
            # (it answers `invalid model name`, which reads like a server fault).
            model = args.student[len("ollama:"):] if args.student.startswith("ollama:") else args.student
            policy = OllamaPolicy(model, temp=args.temp,
                                  num_ctx=args.num_ctx, seed=args.seed)

    out_pairs = open(args.out, "w") if args.out else None
    out_eps = open(args.episodes, "w") if args.episodes else None
    stats = {"instances": 0, "student_solved": 0, "gold_failed": 0,
             "pairs": 0, "no_pairs": 0, "prompt_too_long": 0, "error": 0}
    # The engine keeps the *last* `num_ctx` tokens and says nothing, so an
    # episode whose history is over budget has already lost its system prompt
    # and its issue statement by the time a turn is sampled. ~3.5 chars/token is
    # the tokenizer's own ratio on this corpus; the guard is deliberately
    # conservative because a false "too long" costs one instance and a false
    # "fits" costs a corrupted pair.
    max_chars = args.max_prompt_chars or int(args.num_ctx * 3.5)
    print(f"context budget: {max_chars} chars (~{args.num_ctx} tok)")
    drops = {}

    for inst in insts:
        stats["instances"] += 1
        t0 = time.time()
        try:
            repo_dir = prepare(inst, root=args.root)

            # 1. our model first: if it solves the instance, the instance is
            #    closed to pairs entirely (the "do not unlearn" rule)
            student_ep, solved = None, False
            if policy is not None:
                student_ep = run_repo_episode(inst, policy, root=args.root,
                                              max_steps=args.student_steps)
                if _over_budget(student_ep, max_chars):
                    stats["prompt_too_long"] += 1
                    print(f"  {inst.instance_id}: student history over budget "
                          f"({_chars(student_ep)} chars) — skipping instance")
                    continue
                solved = bool(student_ep["check"].get("passed"))
                if out_eps:
                    out_eps.write(json.dumps({
                        "instance_id": inst.instance_id, "role": "student",
                        "passed": solved, "n_steps": student_ep.get("steps"),
                        "check": student_ep.get("check"),
                        "grade": student_ep.get("grade"),
                        "messages": student_ep.get("messages"),
                    }) + "\n")

            # 2. the gold replay: executed, not assumed
            gp, steps = gold_policy(inst, repo_dir)
            if gp is None:
                print(f"  {inst.instance_id}: gold does not fit {MAX_STEPS} steps, skip")
                stats["gold_failed"] += 1
                continue
            gold_ep = run_repo_episode(inst, gp, root=args.root,
                                       max_steps=len(steps) + 1)
            if _over_budget(gold_ep, max_chars):
                stats["prompt_too_long"] += 1
                print(f"  {inst.instance_id}: gold history over budget "
                      f"({_chars(gold_ep)} chars) — skipping instance")
                continue
            if not gold_ep["check"].get("passed"):
                print(f"  {inst.instance_id}: GOLD REPLAY FAILED "
                      f"(steps={gold_ep.get('steps')}) — refusing to use it")
                stats["gold_failed"] += 1
                continue

            if out_eps:
                out_eps.write(json.dumps({
                    "instance_id": inst.instance_id, "role": "gold",
                    "passed": True, "n_steps": gold_ep.get("steps"),
                    "check": gold_ep.get("check"), "steps": steps,
                }) + "\n")

            if solved:
                stats["student_solved"] += 1
                print(f"  {inst.instance_id}: student SOLVED — no pairs "
                      f"(kept as a positive)")
                continue

            if policy is None:
                print(f"  {inst.instance_id}: gold ok ({len(steps)} steps, "
                      f"{time.time() - t0:.1f}s)")
                continue

            # 3. turn-level pairs over the gold trajectory's states
            pairs, why = pairs_from_trajectory(
                gold_ep["messages"], make_query(policy), tok,
                key=inst.instance_id, max_prompt_tok=args.max_prompt_tok,
                max_turns=args.max_turns)
            for p in pairs:
                p["kind"] = call_kind(p["chosen"])
                p["instance_id"] = inst.instance_id
                if out_pairs:
                    out_pairs.write(json.dumps(p) + "\n")
            for k, v in (why or {}).items():
                drops[k] = drops.get(k, 0) + v
            stats["pairs"] += len(pairs)
            if not pairs:
                stats["no_pairs"] += 1
            print(f"  {inst.instance_id}: student failed, gold ok, "
                  f"{len(pairs)} pairs {why or ''} ({time.time() - t0:.1f}s)")
        except Exception as e:                                   # noqa: BLE001
            stats["error"] += 1
            print(f"  {inst.instance_id}: ERROR {type(e).__name__}: {e}")
            if args.verbose:
                traceback.print_exc()

    for fh in (out_pairs, out_eps):
        if fh:
            fh.close()
    summary = {**stats, "drops": drops, "max_prompt_chars": max_chars}
    print("\n" + json.dumps(summary, indent=1))
    # The kernel's own report only sees the pair file, so the reasons an
    # instance produced nothing (no_signal / long_prompt / prompt_too_long)
    # would otherwise live only in the log.
    report_path = Path(args.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(summary, indent=1))
    return 0 if stats["error"] == 0 else 1


# ---------------------------------------------------------------------------
def selftest() -> int:
    bad = 0
    diff = (
        "diff --git a/src/action.js b/src/action.js\n"
        "index 1111111..2222222 100644\n"
        "--- a/src/action.js\n"
        "+++ b/src/action.js\n"
        "@@ -10,7 +10,8 @@ function run() {\n"
        " const a = 1;\n"
        "-const b = 2;\n"
        "+const b = 3;\n"
        "+const c = 4;\n"
        " return a;\n"
        "@@ -30,3 +31,3 @@ function other() {\n"
        "-  return null;\n"
        "+  return undefined;\n"
        "}\n"
        "diff --git a/src/new.js b/src/new.js\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/src/new.js\n"
        "@@ -0,0 +1,2 @@\n"
        "+module.exports = 1;\n"
        "+// end\n"
    )
    files = parse_patch(diff)
    if [f["path"] for f in files] != ["src/action.js", "src/new.js"]:
        print(f"  FAIL paths: {[f['path'] for f in files]}"); bad += 1
    if len(files[0]["hunks"]) != 2 or len(files[1]["hunks"]) != 1:
        print("  FAIL hunk count"); bad += 1
    if files[0]["hunks"][0]["old"] != ["const a = 1;", "const b = 2;", " return a;".lstrip()]:
        print(f"  FAIL hunk old: {files[0]['hunks'][0]['old']}"); bad += 1
    if files[0]["hunks"][0]["new"][1] != "const b = 3;":
        print(f"  FAIL hunk new: {files[0]['hunks'][0]['new']}"); bad += 1
    if not files[1]["new_file"] or files[1]["hunks"][0]["new"] != ["module.exports = 1;", "// end"]:
        print("  FAIL new file"); bad += 1
    if _block([]) != "" or _block(["a"]) != "a\n":
        print("  FAIL _block"); bad += 1

    if call_kind('x<|tool_call|>{"name": "bash", "arguments": {"command": "ls"}}') != "run":
        print("  FAIL kind run"); bad += 1
    if call_kind('{"name": "str_replace_editor", "arguments": {"command": "view"}}') != "read":
        print("  FAIL kind read"); bad += 1
    if call_kind('{"name": "str_replace_editor", "arguments": {"command": "str_replace"}}') != "edit":
        print("  FAIL kind edit"); bad += 1
    if call_kind('{"name": "submit", "arguments": {}}') != "submit":
        print("  FAIL kind submit"); bad += 1
    if call_kind("no call here") != "none":
        print("  FAIL kind none"); bad += 1

    class _Inst:
        fix_commit, base_commit = "f" * 40, "b" * 40
        source_files = ["src/a.js"]
        test_files = ["src/a.test.js"]
        check_cmd = "npx jest"
    orig = globals()["_git"]

    def fake_git(repo, *a, check=True):
        if a[0] == "diff":
            return diff
        if a[0] == "show":
            return {"b" * 40 + ":src/action.js": "const a = 1;\nconst b = 2;\n return a;\n",
                    "f" * 40 + ":src/action.js": "const a = 1;\nconst b = 3;\nconst c = 4;\n return a;\n",
                    }.get(a[1], "")
        return ""
    globals()["_git"] = fake_git
    try:
        steps = edit_steps(_Inst(), "/nowhere")
    finally:
        globals()["_git"] = orig
    if len(steps) != 2 or '"str_replace"' not in steps[0]:
        print(f"  FAIL edit_steps: {steps}"); bad += 1

    print("selftest:", "ok" if not bad else f"{bad} failure(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default="rl/instances/vault-action.jsonl")
    ap.add_argument("--out", default="rl/samples/real-pairs.jsonl")
    ap.add_argument("--episodes", default="rl/samples/real-episodes.jsonl")
    ap.add_argument("--root", default=None, help="REALENV_ROOT-style scratch dir")
    ap.add_argument("--student", default="tinyballs-dpo",
                    help="Ollama model tag for the rejected side, or "
                         "`hf:<export_dir>` to serve the checkpoint in-process "
                         "(a Kaggle kernel has no Ollama daemon)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="per-turn cap for the in-process (hf:) path")
    ap.add_argument("--student-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-prompt-tok", type=int, default=14000,
                    help="drop a pair whose rendered prompt exceeds this")
    ap.add_argument("--max-prompt-chars", type=int, default=None,
                    help="drop an instance whose episode history exceeds this "
                         "(default: 3.5 x --num-ctx)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json; without it prompt length is not checked")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--gold-only", action="store_true",
                    help="replay the fix only: prove the chosen side, no student")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.episodes:
        Path(a.episodes).parent.mkdir(parents=True, exist_ok=True)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
