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

  1b. **`submit` and the check are additive bonuses, not a gate (§4.2b).** The
     ladder says how good the final file state is; `submit` pays a flat bonus and a
     re-run of the task's own check pays a small one. Rounds 1–4 gated these (see the
     block comment by `R_TERM`): every gate made the policy keep one skill and drop
     the other, and round 4 reached its best headline while scoring 0.0 on its own
     reward. Decoupling lets both live. The do-nothing control is untouched — an
     immediate submit is rung `none`, pays no termination bonus, and scores 0.0
     (§4.5).

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

# the ladder (§4.2) — one place, so grpo_report and the trainer read the same cut.
# Round 5 de-farmed it: `progress` (an edit that *moved* the check output but still
# failed) paid 0.6 and was the cheapest rung to reach, so "one plausible edit and
# leave" was the optimum and the pass rate never moved. It pays 0.2 now — small next
# to a pass (1.0) — but keeps the top partial rung, because a change that moves the
# check is more than a no-op on the right file (`correct_file`, 0.1).
R_PASS = 1.0
R_PROGRESS = 0.2
R_CORRECT_FILE = 0.1
R_EDITED = 0.05
R_NONE = 0.0
# §4.2b (round 5) — the reward is ADDITIVE, not a gate. Rounds 1–4 proved that a
# gate between two independent skills makes the policy keep only one of them:
#   round 1  reward = file state            -> verify 0.79, terminate 0.00
#   round 2  + a +0.2 rung-gated bonus      -> verify 0.87, terminate 0.03  (too weak)
#   round 3  hard gate on `submit`          -> terminate 0.27, verify 0.05
#   round 4  hard gate on `submit ∧ verify` -> terminate 0.80, verify 0.00
# Round 4 is the tell: it hit the best headline yet while scoring 0.0 on its own
# reward — the gate had made the reward unreachable, so GRPO optimized a proxy.
# Solving (`ladder`) and terminating (`submit`) are decoupled additive terms now, so
# the policy cannot trade one for the other. `submit` pays a flat R_TERM, but only
# when the rung is non-zero, so the §4.5 do-nothing submit stays 0.0. A re-run of the
# task's own check pays a small dense R_CHECK: as a *gate* the verbatim, task-specific
# check was an all-or-nothing zero the on-policy gradient could not hold; as a bonus
# it is a gradient the policy can climb every time it emits it.
R_TERM = 0.4
R_CHECK = 0.1

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

    # §4.2b (round 5) — additive and decoupled. `ladder` is what the final file state
    # is worth; a clean termination pays a flat bonus (only off a non-zero rung) and
    # re-running the task's own check pays a small one. None zeroes the others, so the
    # policy can hold "solve" and "terminate" at once (rounds 1–4 each dropped one).
    # §4.5 holds: a do-nothing immediate submit has no edit, so `ladder` is 0 and the
    # termination bonus does not fire. Anti-hack stays 0.0 flat (it never reaches the
    # ladder).
    if anti:
        reward = R_NONE
    else:
        reward = ladder
        if submitted and ladder > 0:
            reward += R_TERM
        if verified:
            reward += R_CHECK

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

    print("== oracle (honest fix, check, submit) -> 1.0 + 0.4 + 0.1 = 1.5")
    ep = grade_run(oracle_policy(task))
    g = ep["grade"]
    oracle_reward = g["reward"]
    ok("oracle_reward", g["reward"] == 1.5, str(g))
    ok("oracle_submitted", ep["submitted"] is True,
       "the oracle earns the §4.2b termination bonus")
    ok("oracle_verified", g["verified"] is True,
       "the oracle earns the §4.2b check bonus")
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

    print("== §4.2b (round 5, additive): a wrong-but-changed edit alone = its rung "
          "(0.2 — no submit, no check, and none of the three zeroes the others)")
    def progress_policy(messages):
        if not progress_policy.done:
            progress_policy.done = True
            return _view()
        return _wrong_edit()
    progress_policy.done = False
    ep_progress = grade_run(progress_policy)
    ok("progress_rung_recorded", ep_progress["grade"]["ladder"] == 0.2,
       str(ep_progress["grade"]))
    ok("progress_no_submit_keeps_ladder", ep_progress["grade"]["reward"] == 0.2,
       "the rung stands alone; nothing is gated away")

    print("== §4.2b (round 5): progress + check + submit = 0.2 + 0.4 + 0.1 = 0.7")
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
    ok("bonuses_stack",
       abs(gok["reward"] - 0.7) < 1e-9 and gok["ladder"] == 0.2, str(gok))
    ok("bonuses_beat_bare_rung", gok["reward"] > ep_progress["grade"]["reward"],
       f"{gok['reward']} vs {ep_progress['grade']['reward']}")

    print("== §4.2b (round 5): submit without re-running the check still banks its "
          "rung + termination bonus = 0.2 + 0.4 = 0.6 (round 4 zeroed all of this)")
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
    ok("submit_without_check_still_banks",
       abs(gnv["reward"] - 0.6) < 1e-9 and gnv["ladder"] == 0.2, str(gnv))

    print("== §4.2b (round 5): re-running the check pays even without submit "
          "(the dense signal that makes the verbatim check learnable)")
    def check_no_submit(messages):
        n = check_no_submit.n
        check_no_submit.n += 1
        if n == 0:
            return _view()
        if n == 1:
            return _wrong_edit()
        if n == 2:
            return _bash(task.check_cmd)
        return {"text": "still failing, I think.", "stopped": True, "calls": []}
    check_no_submit.n = 0
    gcn = grade_run(check_no_submit)["grade"]
    ok("check_bonus_without_submit",
       gcn["verified"] is True and gcn["submitted"] is False
       and abs(gcn["reward"] - 0.3) < 1e-9, str(gcn))

    print("== §4.2b (round 5): a *passing* fix that never submits keeps the pass "
          "rung (1.0) but forgoes the termination bonus (1.5 -> 1.0)")
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
    ok("pass_no_submit_keeps_ladder", gns["reward"] == 1.0 and gns["tier"] == "pass",
       str(gns))
    ok("terminate_is_worth_more_when_solved", oracle_reward > gns["reward"],
       f"oracle {oracle_reward} vs timeout {gns['reward']}")

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
