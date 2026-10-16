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

  1b. **Termination bonus (§4.2b).** The ladder paid for work but never for
     `submit`, and stopping later is worth more rungs — so GRPO round 1 learned to
     never submit (G4: terminate 71.5% -> 0%). A submitted episode that earned a
     nonzero rung now gets `+R_SUBMIT` (capped at `R_PASS`); the do-nothing control
     still scores 0.0, because the bonus is gated on the rung.

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
# §4.2b — termination bonus. `submit` is an explicit action the policy *can* take and
# Stage-A *did* take (58% before the §4.4 prompt, 71.5% under it), but the ladder
# never rewarded it, while *not* submitting buys more steps and so more chances to
# climb a rung. GRPO round 1 optimised that: terminate 71.5% -> 0% (G4). This bonus
# pays for terminating, but only on top of a *nonzero* rung — gated so the §4.5
# do-nothing control (immediate `submit`) still scores 0.0, which is the whole point
# of that control. The total is capped at R_PASS so `pass` stays the unique top and
# every existing `reward == 1.0` check (grpo_report, pass_rate) still means "passed".
# Weight is a starting point like the rest of the ladder (§14) — tune from G0.
R_SUBMIT = 0.2

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
    rbe = _read_before_edit((episode or {}).get("messages"))
    submitted = bool((episode or {}).get("submitted"))

    if anti:
        reward, tier = R_NONE, "anti_hack"
    elif visible["passed"]:
        reward, tier = R_PASS, "pass"
    elif progressed:
        reward, tier = R_PROGRESS, "progress"
    elif correct_file:
        reward, tier = R_CORRECT_FILE, "correct_file"
    elif edited and rbe:
        reward, tier = R_EDITED, "edited_read_first"
    else:
        reward, tier = R_NONE, "none"

    # §4.2b: pay for terminating, but only on top of real work (a nonzero rung), and
    # never above `pass`. An immediate/do-nothing submit has rung `none` (0.0) and so
    # still scores 0.0 (§4.5).
    submit_bonus = R_SUBMIT if (submitted and reward > R_NONE) else 0.0
    reward = min(R_PASS, reward + submit_bonus)

    return {
        "reward": reward,
        "tier": tier,
        "submitted": submitted,
        "submit_bonus": submit_bonus,
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
       "the oracle submits; §4.2b caps it back to 1.0 rather than 1.2")
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

    print("== ladder: a wrong-but-changed source edit -> progress (0.6)")
    def progress_policy(messages):
        if not progress_policy.done:
            progress_policy.done = True
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "view", "path": path}}), "stopped": True, "calls": []}
        # change the constant to a *different* wrong value: check still fails, but
        # the failure output changed, which is the 0.6 rung
        wrong = task.fix["old_str"].split("+")[1].strip()
        return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
            "command": "str_replace", "path": path,
            "old_str": src.splitlines()[1].strip(),
            "new_str": f"return x + {int(wrong) + 1000}"}}), "stopped": True, "calls": []}
    progress_policy.done = False
    ep_progress = grade_run(progress_policy)
    ok("progress_reward", ep_progress["grade"]["reward"] == 0.6, str(ep_progress["grade"]))

    print("== §4.2b: the *same* progress, but it submits, scores higher; and pass "
          "still caps at 1.0")
    def progress_submit_policy(messages):
        n = progress_submit_policy.n
        progress_submit_policy.n += 1
        if n == 0:
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "view", "path": path}}), "stopped": True, "calls": []}
        if n == 1:
            wrong = task.fix["old_str"].split("+")[1].strip()
            return {"text": tc + json.dumps({"name": "str_replace_editor", "arguments": {
                "command": "str_replace", "path": path,
                "old_str": src.splitlines()[1].strip(),
                "new_str": f"return x + {int(wrong) + 1000}"}}), "stopped": True, "calls": []}
        return {"text": tc + json.dumps({"name": "submit", "arguments": {}}),
                "stopped": True, "calls": []}
    progress_submit_policy.n = 0
    ep_psub = grade_run(progress_submit_policy)
    gsub = ep_psub["grade"]
    ok("submit_episode_submitted", ep_psub["submitted"] is True)
    ok("submit_bonus_applied", gsub["reward"] == 0.8 and gsub["submit_bonus"] == 0.2,
       str(gsub))
    ok("submit_outscores_nosubmit",
       gsub["reward"] > ep_progress["grade"]["reward"],
       f"{gsub['reward']} vs {ep_progress['grade']['reward']}")

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
