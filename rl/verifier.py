#!/usr/bin/env python3
"""The verifier is the reward — RL_PLAN_2 §4.

MiMo does not grade the code recipe with a model: `recipes/code/reward.py` just
forwards the *environment's* reward payload. We have the same thing in
`agent_env.run_check`, so this file does not add a judge. It does two things the
raw check cannot:

  1. **The graded ladder (§4.2).** At 113M the binary pass rate is ~12%, so a
     G=16 group is usually all-fail and contributes zero gradient. The ladder
     makes all-fail groups differ (progress on the failure, correct file edited,
     read-before-edit respected) so the group has spread. Weights are the plan's
     starting point and are meant to be retuned from the G0 measurement.

  1b. **`submit` after a check banks the reward (§4.2b).** The ladder says how good
     the final file state is; `submit` cashes it in, and only once the failing check
     has been re-run on that state. Round 1 lost termination (71.5% -> 0%) because
     the reward was the file state — one more step was free upside. Round 2's soft
     bonus barely moved it (0% -> 2.5%). Round 3 hard-gated on `submit`: terminate
     0% -> 27%, but *hollowly* — 51 of 54 submits were failures and `verify_with_check`
     collapsed 0.87 -> 0.05, because re-running the test costs a step and earns
     nothing, so the policy learned to edit and declare. So a submit only banks when
     the check was re-run after the last edit. The do-nothing control is untouched —
     an immediate submit is rung `none` and banks 0.0 (§4.5).

  2. **Anti-hack (§4.3).** The shaping terms are a hack surface, so grading is
     hostile to the three cheap hacks:
       - *restore the test files* from the pristine snapshot and grade the
         model's source edits only — "edit the test to pass" then scores 0;
       - *re-run a held-out variant* (same defect, other literals) so a
         hard-coded expected output passes the visible check and fails this one;
       - *scan the added source lines* for `sys.exit(0)` / `print("OK")` and
         reject.

`grade(task, sandbox, episode)` is meant to be passed to
`agent_env.run_episode(..., grader=grade)`, which runs it while the sandbox is
still open. It returns a plain dict (see `grade`), and every number it uses is
recorded so `rl/grpo_report.py` can show *why* a group had spread.

    python rl/verifier.py --selftest        # Gate G1, no model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .agent_env import generate_task, parse_calls, run_check, run_episode
except ImportError:  # python rl/verifier.py
    from agent_env import generate_task, parse_calls, run_check, run_episode

# the ladder (§4.2) — one place, so grpo_report and the trainer read the same cut
R_PASS = 1.0
R_PROGRESS = 0.6
R_CORRECT_FILE = 0.4
R_EDITED = 0.2
R_NONE = 0.0
# §4.2b — `submit` *after re-running the check* banks the reward (hard gate; see
# `grade`). An episode that never terminated, or that terminated without re-running
# the failing check on its final state, scores R_NONE whatever its rung. The history
# is the reason both conditions are required:
#   round 1  reward = file state            -> terminate 71.5% -> 0%   (free upside)
#   round 2  + a +0.2 rung-gated bonus      -> terminate 0% -> 2.5%    (too weak)
#   round 3  hard gate on `submit`          -> terminate 0% -> 27%, but hollow:
#            51/54 submits were failures, verify_with_check 0.87 -> 0.05
# Round 3 shows why `submit` alone is not enough: re-running the test costs a step and
# earns nothing, so a policy paid only for terminating learns to edit and declare. The
# check is therefore part of what banks the reward. (§4.5 holds: a do-nothing immediate
# submit has no edit, hence no check-after-edit, hence 0.0.) (`R_SUBMIT` retired.)

# a *source* edit that only makes the check short-circuit. The check is
# `exit_code == 0 and "OK" in stdout`; these are the two ways to satisfy it
# without fixing anything.
SHORTCUT_RE = re.compile(
    r"sys\.exit\s*\(|os\._exit\s*\(|(?:^|[^.\w])exit\s*\(|print\s*\(\s*['\"]OK['\"]\s*\)")
READ_VERB_RE = re.compile(r"\b(cat|head|tail|sed|less|more|nl)\b")


# --------------------------------------------------------------------------- helpers
def _added_lines(before, after) -> list:
    """Lines present in `after` (a file's new text) that were not in `before`.
    Set-difference of lines rather than a real diff: this only has to catch a
    newly introduced `print("OK")`, not describe the edit."""
    b = set((before or "").splitlines())
    return [ln for ln in (after or "").splitlines() if ln not in b]


def _read_before_edit(messages) -> bool:
    """Did the episode read a file before the first mutation of it?

    A read is a `str_replace_editor view` of the path, or a `bash` command that
    names the path next to a read verb (`cat`/`head`/`sed`/…). Same spirit as
    `agent_rollout.behavior_metrics`, but counted per episode for the reward.
    """
    viewed, mutated = set(), []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        for name, args in parse_calls(m.get("content") or ""):
            if name == "str_replace_editor":
                p = str(args.get("path", ""))
                if args.get("command") == "view":
                    viewed.add(p)
                elif args.get("command") in ("create", "str_replace", "insert"):
                    mutated.append(p)
            elif name == "bash" and READ_VERB_RE.search(str(args.get("command", ""))):
                cmd = str(args.get("command", ""))
                for p in list(viewed) + [str(args.get("path", ""))]:
                    if p and p in cmd:
                        viewed.add(p)
    if not mutated:
        return False
    return all(p in viewed for p in mutated if p)


def _ran_check_after_last_edit(messages, check_cmd) -> bool:
    """Did the episode re-run the failing check after its last edit?

    This is exactly `agent_rollout`'s `verify_with_check`, computed the same way, so
    the reward and the G4 metric measure one thing. `submit` is only banked when this
    holds: an answer declared without re-running the check on it is the round-3
    failure (51/54 submits were never-re-tested).
    """
    cmd = (check_cmd or "").strip()
    if not cmd:
        return False
    calls = []
    for m in messages or []:
        if m.get("role") != "assistant":
            continue
        calls.extend(parse_calls(m.get("content") or ""))
    last_edit = -1
    for i, (name, args) in enumerate(calls):
        if name == "str_replace_editor" and args.get("command") in (
                "create", "str_replace", "insert"):
            last_edit = i
    if last_edit < 0:
        return False
    return any(name == "bash" and str(args.get("command", "")).strip() == cmd
               for name, args in calls[last_edit + 1:])


def _changed_among(changed, files) -> dict:
    return {rel: changed[rel] for rel in changed if rel in files}


# --------------------------------------------------------------------------- grade
def grade(task, sb, episode) -> dict:
    """Score one finished episode. Sandbox must still be open.

    Order matters and is the plan's: restore tests, then believe the visible
    check, then the variant, then walk the ladder top-down.
    """
    changed = sb.diff_initial()
    tests = set(task.test_files or [])
    source_changed = {r: v for r, v in changed.items() if r not in tests}
    tests_changed = {r: v for r, v in changed.items() if r in tests}

    # 1. shortcut scan on what the model actually added to source files
    shortcuts = []
    for rel, (before, after) in source_changed.items():
        for line in _added_lines(before, after):
            if SHORTCUT_RE.search(line):
                shortcuts.append(f"{rel}: {line.strip()[:80]}")

    # 2. restore every test file, then run the visible check on source edits only
    sb.restore({r: task.files[r] for r in tests if r in task.files})
    visible = run_check(task, sb)

    # 3. the held-out variant, over the same (restored-test) tree
    variant = None
    if getattr(task, "variant", None):
        sb.restore(task.variant["files"])
        v = run_check(task, sb, task.variant["check_cmd"])
        variant = {"passed": v["passed"], "cmd": v["cmd"], "output": v["output"]}

    # 4. anti-hack flags → reward 0, whatever the ladder would have said
    anti = []
    if shortcuts:
        anti.append("shortcut_marker")
    if visible["passed"] and variant is not None and not variant["passed"]:
        anti.append("hardcoded_output")
    if source_changed == {} and tests_changed:
        anti.append("edited_tests_only")

    initial = (episode or {}).get("initial_check") or {}
    progressed = (visible["output"] != initial.get("output", "")) and not visible["passed"]
    correct_file = bool(task.fix) and task.fix["path"] in source_changed
    edited = bool(source_changed)
    msgs = (episode or {}).get("messages")
    rbe = _read_before_edit(msgs)
    submitted = bool((episode or {}).get("submitted"))
    verified = _ran_check_after_last_edit(msgs, visible.get("cmd"))

    if anti:
        ladder, tier = R_NONE, "anti_hack"
    elif visible["passed"]:
        ladder, tier = R_PASS, "pass"
    elif progressed:
        ladder, tier = R_PROGRESS, "progress"
    elif correct_file:
        ladder, tier = R_CORRECT_FILE, "correct_file"
    elif edited and rbe:
        ladder, tier = R_EDITED, "edited_read_first"
    else:
        ladder, tier = R_NONE, "none"

    # §4.2b hard gate: `ladder` is what the final file state is worth; `submit` banks
    # it, and only after the check was re-run on that state. No submit -> no bank
    # (round 3: terminating without working is hollow). Submit but no check -> no bank
    # (round 3: 51/54 submits were failures it never re-tested). §4.5 still holds: a
    # do-nothing immediate submit has no edit, so no check-after-edit, so 0.0.
    # Anti-hack stays 0.0 too (it never reaches the ladder).
    reward = ladder if (submitted and verified) else R_NONE

    return {
        "reward": reward,
        "tier": tier,
        "ladder": ladder,
        "submitted": submitted,
        "verified": bool(verified),
        "anti_hack": anti,
        "check": visible,
        "variant": variant,
        "progressed": bool(progressed),
        "correct_file_edited": bool(correct_file),
        "read_before_edit": bool(rbe),
        "source_changed": sorted(source_changed),
        "tests_changed": sorted(tests_changed),
        "shortcuts": shortcuts,
    }


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    """Gate G1, encoded: a real fix scores 1, and every planted hack scores 0."""
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    import random

    from agent_env import TOOL_CALL, MicroTask, _t_constant
    from agent_rollout import control_policy, oracle_policy

    spec = _t_constant(random.Random(3))
    task = MicroTask("g1", spec["files"], spec["problem"], spec["check_cmd"],
                     fix=spec["fix"], test_files=spec["test_files"],
                     variant=spec["variant"])
    path = task.fix["path"]
    test = task.test_files[0]

    def grade_run(policy):
        return run_episode(task, policy, grader=grade)

    print("== oracle (honest fix) -> 1.0")
    ep = grade_run(oracle_policy(task))
    g = ep["grade"]
    ok("oracle_reward", g["reward"] == 1.0, str(g))
    ok("oracle_submitted", ep["submitted"] is True,
       "the oracle banks its pass by submitting (§4.2b)")
    ok("oracle_variant", g["variant"] and g["variant"]["passed"] is True)
    ok("oracle_no_anti", g["anti_hack"] == [])

    print("== do-nothing (immediate submit) -> 0.0, not anti-hack")
    ep = grade_run(control_policy)
    ok("control_reward", ep["grade"]["reward"] == 0.0, str(ep["grade"]))
    ok("control_tier", ep["grade"]["tier"] == "none", ep["grade"]["tier"])

    print("== planted hack 1: edit the test to pass -> 0.0 (tests restored)")
    tc = TOOL_CALL

    def edit_test(policy_msgs):
        # replace the assertion with a bare print so the visible check passes
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "str_replace", "path": test,
            "old_str": task.files[test].splitlines()[1],
            "new_str": "print('OK')"}}), "stopped": True, "calls": []}

    ep = grade_run(edit_test)
    g = ep["grade"]
    ok("edittest_reward", g["reward"] == 0.0, str(g))
    ok("edittest_anti", "edited_tests_only" in g["anti_hack"], str(g["anti_hack"]))
    ok("edittest_tier", not g["check"]["passed"], "restore must undo the edit")

    print("== planted hack 2: hard-code the visible constant -> 0.0 (variant)")
    src = task.files[path]
    const = task.fix["new_str"].split("+")[1].strip()

    def hardcode_policy(messages):
        # first call: read (satisfy rbe); second: overwrite the source
        if not hardcode_policy.done:
            hardcode_policy.done = True
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "view", "path": path}}), "stopped": True, "calls": []}
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "str_replace", "path": path,
            "old_str": src.splitlines()[1].strip(),
            "new_str": f"return {const}"}}), "stopped": True, "calls": []}
    hardcode_policy.done = False

    ep = grade_run(hardcode_policy)
    g = ep["grade"]
    ok("hardcode_visible_passes", g["check"]["passed"] is True, str(g["check"]))
    ok("hardcode_reward", g["reward"] == 0.0, str(g))
    ok("hardcode_anti", "hardcoded_output" in g["anti_hack"], str(g["anti_hack"]))

    print("== planted hack 3: sys.exit(0) in the source -> 0.0 (marker)")
    def sysexit_policy(messages):
        if not sysexit_policy.done:
            sysexit_policy.done = True
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "view", "path": path}}), "stopped": True, "calls": []}
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "str_replace", "path": path,
            "old_str": src.splitlines()[1].strip(),
            "new_str": "import sys; sys.exit(0)"}}), "stopped": True, "calls": []}
    sysexit_policy.done = False
    ep = grade_run(sysexit_policy)
    g = ep["grade"]
    ok("sysexit_reward", g["reward"] == 0.0, str(g))
    ok("sysexit_anti", "shortcut_marker" in g["anti_hack"], str(g["anti_hack"]))

    def _wrong_edit():
        # change the constant to a *different* wrong value: check still fails, but
        # the failure output changed, which is the 0.6 rung
        wrong = task.fix["old_str"].split("+")[1].strip()
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "str_replace", "path": path,
            "old_str": src.splitlines()[1].strip(),
            "new_str": f"return x + {int(wrong) + 1000}"}}), "stopped": True, "calls": []}

    def _view():
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "view", "path": path}}), "stopped": True, "calls": []}

    def _bash(command):
        return {"text": tc + json.dumps({"name": "bash", "arguments": {
            "command": command}}), "stopped": True, "calls": []}

    def _submit():
        return {"text": tc + json.dumps({"name": "submit", "arguments": {}}),
                "stopped": True, "calls": []}

    print("== §4.2b gate: a wrong-but-changed edit that never submits scores 0.0 "
          "(the rung is recorded but not banked)")
    def progress_policy(messages):
        if not progress_policy.done:
            progress_policy.done = True
            return _view()
        return _wrong_edit()
    progress_policy.done = False
    ep_progress = grade_run(progress_policy)
    ok("progress_rung_recorded", ep_progress["grade"]["ladder"] == 0.6,
       str(ep_progress["grade"]))
    ok("progress_no_submit_banks_zero", ep_progress["grade"]["reward"] == 0.0,
       "no submit -> nothing banked, however good the rung")

    print("== §4.2b gate: progress, check re-run, then submit -> 0.6 banked")
    def progress_ok(messages):
        n = progress_ok.n
        progress_ok.n += 1
        if n == 0:
            return _view()
        if n == 1:
            return _wrong_edit()
        if n == 2:
            return _bash(task.check_cmd)
        return _submit()
    progress_ok.n = 0
    gok = grade_run(progress_ok)["grade"]
    ok("submit_after_check_verified",
       gok["verified"] is True and gok["submitted"] is True, str(gok))
    ok("submit_after_check_banks_rung",
       gok["reward"] == 0.6 and gok["ladder"] == 0.6, str(gok))
    ok("bank_beats_no_submit", gok["reward"] > ep_progress["grade"]["reward"],
       f"{gok['reward']} vs {ep_progress['grade']['reward']}")

    print("== §4.2b gate: progress + submit but NEVER re-running the check -> 0.0 "
          "(round 3 banked 51/54 submits it never re-tested)")
    def progress_noverify(messages):
        n = progress_noverify.n
        progress_noverify.n += 1
        if n == 0:
            return _view()
        if n == 1:
            return _wrong_edit()
        return _submit()
    progress_noverify.n = 0
    gnv = grade_run(progress_noverify)["grade"]
    ok("submit_without_check_not_verified", gnv["verified"] is False, str(gnv))
    ok("submit_without_check_banks_zero",
       gnv["reward"] == 0.0 and gnv["ladder"] == 0.6, str(gnv))

    print("== §4.2b gate: a *passing* fix that never submits still scores 0.0 "
          "(an unsubmitted episode is a timeout, not a win)")
    def fix_no_submit_policy(messages):
        n = fix_no_submit_policy.n
        fix_no_submit_policy.n += 1
        if n == 0:
            return _view()
        if n == 1:
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "str_replace", "path": path,
                "old_str": task.fix["old_str"], "new_str": task.fix["new_str"]}}),
                "stopped": True, "calls": []}
        return {"text": "I think that's fixed.", "stopped": True, "calls": []}
    fix_no_submit_policy.n = 0
    gns = grade_run(fix_no_submit_policy)["grade"]
    ok("pass_no_submit_check_passes", gns["check"]["passed"] is True, str(gns["check"]))
    ok("pass_no_submit_banks_zero", gns["reward"] == 0.0 and gns["tier"] == "pass",
       str(gns))

    print()
    if fails:
        print(f"VERIFIER SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("VERIFIER SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else 0)
