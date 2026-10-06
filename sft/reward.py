"""Shaped reward for TinyBalls-V1 RL — the gate on Track 1 (`RL_TRAINING.md` §1).

    total = 0.2 * format + 0.3 * declared + 0.3 * grounded + 0.2 * correct

Why shaped rather than sparse: `multi_tool` sits at 0/8 task-correct through the
unified serving path, so an unshaped reward is constant across a rollout group,
every group-relative advantage is 0, and GRPO completes a run with a moving loss
and no gradient. The shaped terms score ~0.8-1.0 on that scenario already, which
is what gives the group a spread to learn from.

The weights are `TOOL_FIX_PLAN.md` §7's, unchanged. The *target* is the answer,
not the calls: on multi_tool the model already issues both correct calls with
correct arguments 8/8 (`REPAIR2_EVAL.md` §3), so gradient spent on declared
names and argument keys is gradient spent on a solved problem.

Env is `sft/tinyballs_env.py`, the same mock the harness and the error-recovery
training rows were generated from, fault injection included. Grading semantics
for `correct` follow `sft/repair-eval/passk_grade.py`.

Self-test (no model, no GPU):

    python sft/reward.py --selftest
"""

import argparse
import json
import re

from tinyballs_env import FaultState, PEOPLE, WEATHER, run_tool

WEIGHTS = {"format": 0.2, "declared": 0.3, "grounded": 0.3, "correct": 0.2}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city. Valid cities: Tokyo, Paris, London, New York.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "lookup_person",
        "description": "Look up facts about a famous person in a database.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '12 + 3 * 4'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]

DECLARED = {"get_weather": "city", "lookup_person": "name", "calculate": "expression"}
IM_START = "<" + "|im_start|>"
IM_END = "<" + "|im_end|>"
TOOL_CALL = "<" + "|tool_call|>"
TOOL_RESULT = "<" + "|tool_result|>"


def tools_system(tools=None):
    """The training system prompt, byte for byte (normalize_sft.tools_system)."""
    tools = tools or TOOLS
    head = ("You are a helpful assistant with access to tools. Use the provided tools "
            "when they can help answer the user's request. If a tool call is needed, "
            "put any reasoning in <think>...</think> first.\nAvailable tools:\n")
    tail = ('\nTo call a tool, reply with ' + TOOL_CALL
            + '{"name": "...", "arguments": {...}} and nothing after it.')
    return head + json.dumps(tools, ensure_ascii=False) + tail


# ------------------------------------------------------------------ tasks
class Task:
    """One RL prompt plus the facts its answer has to carry.

    `entities` maps a name the answer must mention to the value it must bind to
    it — that mapping is the whole point of Track 1: on multi_tool the model
    names both cities but attaches 18.4 to Tokyo and drops or swaps the other.
    `derived` is an optional arithmetic check on top (22.1 - 18.4 = 3.7).
    """

    def __init__(self, name, prompt, calls=(), entities=None, derived=None,
                 faults=(), must_call=()):
        self.name = name
        self.prompt = prompt
        self.calls = list(calls)          # (tool, args) pairs a correct run makes
        self.entities = dict(entities or {})
        self.derived = derived            # {"label": str, "value": float}
        self.faults = tuple(faults)
        self.must_call = tuple(must_call)  # (tool, args) that must appear at least once

    def messages(self):
        return [{"role": "system", "content": tools_system()},
                {"role": "user", "content": self.prompt}]


def compare_weather_task(a="Tokyo", b="New York", faults=("transient",)):
    """The multi_tool shape, parameterised so RL cannot memorise one prompt."""
    ta, tb = WEATHER[a]["temp_c"], WEATHER[b]["temp_c"]
    diff = abs(tb - ta)
    warmer = b if tb > ta else a
    return Task(
        name=f"compare_weather_{a}_{b}".replace(" ", "_"),
        prompt=(f"Compare the current temperature in {a} and {b}. "
                f"Which one is warmer and by how many degrees?"),
        calls=[("get_weather", {"city": a}), ("get_weather", {"city": b})],
        must_call=[("get_weather", {"city": a}), ("get_weather", {"city": b})],
        entities={a: f"{ta}", b: f"{tb}"},
        derived={"label": f"{round(diff, 1)}", "value": diff, "warmer": warmer},
        faults=faults,
    )


def single_weather_task(city="Tokyo"):
    t = WEATHER[city]["temp_c"]
    return Task(name=f"single_weather_{city}".replace(" ", "_"),
                prompt=f"What's the current temperature in {city}?",
                calls=[("get_weather", {"city": city})],
                must_call=[("get_weather", {"city": city})],
                entities={city: f"{t}"})


def error_recovery_task(city="Berlin"):
    """Berlin is deliberately absent from the mock DB, so the correct behaviour is
    to report that rather than invent a temperature (When2Call-style abstention)."""
    return Task(name=f"error_recovery_{city}",
                prompt=f"What's the weather like in {city} right now?",
                calls=[("get_weather", {"city": city})],
                must_call=[("get_weather", {"city": city})],
                entities={})


TASKS = [compare_weather_task(), single_weather_task(), error_recovery_task()]


# ------------------------------------------------------------------ parsing
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
        except Exception:
            i = j + 1


def parse_calls(text):
    """Trained-format calls: `<|tool_call|>{...}`, one per segment.

    Tolerates the marker being absent (the Ollama path strips it) because the
    JSON object is still unambiguous, but counts segments separately so the
    format term can see how many blocks a turn had.
    """
    calls = []
    for obj in _json_objects(text):
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) \
                and (obj.get("arguments") is not None or obj.get("parameters") is not None):
            args = obj.get("arguments", obj.get("parameters"))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append((obj["name"], args))
    return calls


# ------------------------------------------------------------------ components
def score_format(turns, stopped=True):
    """0..1. Turn hygiene: closed on im_end, no leaked im_start, one call block
    per turn, every call block valid JSON."""
    if not turns:
        return 0.0
    parts = []
    for text, stop in zip(turns, stopped):
        s = 1.0
        if not stop:
            s -= 0.4                      # ran to the token cap
        if IM_START in text:
            s -= 0.3                      # leaked a turn marker into its own turn
        n_blocks = text.count(TOOL_CALL)
        if n_blocks > 1:
            s -= 0.2 * (n_blocks - 1)     # one call block per turn
        if n_blocks and len(parse_calls(text)) != n_blocks:
            s -= 0.3                      # a block that does not parse
        parts.append(max(0.0, s))
    return sum(parts) / len(parts)


def score_declared(calls):
    """0..1. Fraction of calls using a declared name *and* filling the declared
    argument key. An invented name scores 0 for that call, not a penalty below 0:
    a run that calls the wrong tool three times should not beat one that calls
    it once."""
    if not calls:
        return 0.0
    good = 0
    for name, args in calls:
        key = DECLARED.get(name)
        if key and isinstance(args, dict) and isinstance(args.get(key), str) \
                and args[key].strip():
            good += 1
    return good / len(calls)


def _entities_from_results(results):
    """(entity, value) pairs the env actually returned, so grounding is checked
    against observations rather than against the prompt."""
    pairs = []
    for res in results or []:
        if not isinstance(res, dict) or res.get("error"):
            continue
        if "temp" in res or "temp_c" in res:
            city = res.get("city")
            val = res.get("temp", res.get("temp_c"))
            if city:
                pairs.append((str(city), re.search(r"-?\d+(?:\.\d+)?", str(val)).group(0)))
        if "height_cm" in res and res.get("name"):
            pairs.append((str(res["name"]), str(res["height_cm"])))
    return pairs


def score_grounded(final, results):
    """0..1. Every returned entity is named in the answer *and* its own value is
    attached to it.

    Two ways to count as bound, because the model's two real failure shapes need
    different tests:

    clause      the value sits between this entity's mention and the next
                entity's mention — "Tokyo is 18.4C and New York is 22.1C"
    monotone    the names and the values appear as two aligned lists in the same
                order — "New York is warmer than Tokyo (22.1C vs 18.4C)", where
                the values land after both names and a clause test would
                false-negative a correct answer

    A fixed proximity window was tried first and rejected: in "Tokyo is 22.1
    degrees and New York is 18.4 degrees" both values sit within 60 characters
    of both names, so it scores 1.0 on exactly the mis-binding this track exists
    to fix. Naming both facts somewhere in the answer is what the model already
    does (8/8 mech on multi_tool), so a bag-of-facts check teaches nothing.
    """
    pairs = _entities_from_results(results)
    if not pairs:
        return 0.0
    text = (final or "").lower()

    # clause hits: cut each entity's span at the next entity mention
    names = sorted({e.lower() for e, _ in pairs})
    marks = sorted((m.start(), n) for n in names
                   for m in re.finditer(re.escape(n), text))
    clause = set()
    for entity, value in pairs:
        en = entity.lower()
        for i, (pos, n) in enumerate(marks):
            if n != en:
                continue
            end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
            if value.lower() in text[pos:end]:
                clause.add(en)
                break

    # monotone pairing: k-th first-mentioned entity with the k-th first-printed
    # value, accepted only when every name precedes its own value
    aligned = set()
    ent_first = sorted(((text.find(e.lower()), e.lower()) for e, _ in pairs),
                       key=lambda t: t[0])
    val_first = sorted(((text.find(v.lower()), v) for _, v in pairs),
                       key=lambda t: t[0])
    if (len(ent_first) == len(val_first)
            and all(p >= 0 for p, _ in ent_first)
            and all(p >= 0 for p, _ in val_first)
            and all(ent_first[i][0] < val_first[i][0] for i in range(len(ent_first)))):
        by_entity = {e.lower(): v for e, v in pairs}
        for (_, en), (_, val) in zip(ent_first, val_first):
            if by_entity.get(en) == val:
                aligned.add(en)

    bound = clause | aligned
    return len(bound) / len(pairs)


def score_correct(task, calls, results, final):
    """0..1. Task-level: the required calls were made, the comparison or the
    abstention is present, and the arithmetic is right."""
    if not (final or "").strip():
        return 0.0
    fl = final.lower()
    for tool, args in task.must_call:
        want = {k: str(v).strip().lower() for k, v in args.items()}
        if not any(n == tool and all(str(a.get(k, "")).strip().lower() == v
                                     for k, v in want.items())
                   for n, a in calls):
            return 0.0
    if task.derived:
        # the derived quantity has to be stated, and stated correctly
        label = task.derived["label"]
        if label not in fl and str(round(task.derived["value"], 1)) not in fl:
            return 0.0
        warmer = task.derived["warmer"].lower()
        if warmer not in fl:
            return 0.0
        # a wrong difference stated confidently is worse than none
        for wrong in _wrong_differences(task.derived["value"]):
            if re.search(rf"\b{re.escape(wrong)}\b", fl):
                return 0.0
        return 1.0
    if task.entities:
        # single-tool / lookup: the value must be present
        vals = [v for v in task.entities.values()]
        return 1.0 if all(v in fl for v in vals) else 0.0
    # no expected facts (error recovery): credibly report the failure
    if not any(isinstance(r, dict) and r.get("error") for r in results or []):
        return 0.0
    return 1.0 if re.search(r"unknown|not (available|in|supported|found)|no data|"
                            r"cannot|can't|couldn't|denied|unavailable", fl) else 0.0


def _wrong_differences(value, span=3.0):
    """Plausible wrong differences around the right one, for the "stated a
    confidently wrong number" check. Values inside 0.05 of the truth are the
    right answer at one decimal and must not be listed."""
    out = []
    steps = int(span * 2)
    for i in range(-steps, steps + 1):
        v = round(value + i * 0.5, 1)
        if abs(v) > 0.05 and abs(v - value) > 0.05:
            out.append(str(v))
    return out


# ------------------------------------------------------------------ episode
def score_episode(task, turns, stopped=None, results=None, calls=None):
    """Score one rollout.

    turns    : assistant texts, in order (the raw generations)
    stopped  : per-turn bool, True when the turn ended at im_end
    results  : tool observations, in call order
    calls    : parsed (name, args) per turn; parsed from turns when omitted
    """
    turns = list(turns)
    if stopped is None:
        stopped = [True] * len(turns)
    if calls is None:
        calls = [c for t in turns for c in parse_calls(t)]
    calls = list(calls)
    results = list(results or [])
    final = turns[-1] if turns and not parse_calls(turns[-1]) else ""

    parts = {
        "format": score_format(turns, stopped),
        "declared": score_declared(calls),
        "grounded": score_grounded(final, results),
        "correct": score_correct(task, calls, results, final),
    }
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    # No tool call on a task that needs one scores zero, whatever the prose.
    # A clean "I don't know." still earns the 0.2 format term otherwise, and the
    # AWM do-nothing control is exactly how that goes unnoticed: 1 task in 12
    # "passed" with the agent having done nothing (RL_TRAINING.md §5). Caveat:
    # a track whose correct behaviour is to abstain must leave `must_call` empty
    # for that task, or this gate punishes the right answer.
    if task.must_call and not calls:
        total = 0.0
        parts["no_tool_call"] = 1.0
    return {"total": total, "parts": parts, "n_calls": len(calls),
            "final": final}


def run_episode(task, respond, max_steps=6, faults=None):
    """Drive the env against a `respond(messages) -> (text, stopped)` policy.

    Kept separate from scoring so the same loop serves the variance gate (real
    model), the unit tests (scripted policy) and later the RL env.
    """
    faults = FaultState(list(task.faults if faults is None else faults))
    messages = task.messages()
    turns, stopped, calls, results = [], [], [], []
    for _ in range(max_steps):
        text, stop = respond(messages)
        turns.append(text)
        stopped.append(stop)
        step_calls = parse_calls(text)
        if not step_calls:
            break
        hist = text.strip() or "\n".join(
            TOOL_CALL + json.dumps({"name": n, "arguments": a}, ensure_ascii=False)
            for n, a in step_calls)
        messages.append({"role": "assistant", "content": hist})
        for name, args in step_calls:
            calls.append((name, args))
            fault = faults.take(name, args)
            res = fault if fault is not None else run_tool(name, args)
            results.append(res)
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(res)})
    return score_episode(task, turns, stopped, results, calls), {
        "turns": turns, "calls": calls, "results": results, "messages": messages}


# ------------------------------------------------------------------ self-test
def _policy_script(script):
    """respond() that walks a fixed list of (text, stopped) per step."""
    steps = list(script)

    def respond(messages):
        return steps.pop(0) if steps else ("", True)
    return respond


def selftest():
    """Hand-built episodes with known component values. If these drift, the
    reward is measuring something other than what the plan says."""
    fails = []

    def expect(name, got, want, msg="", tol=1e-6):
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        tail = f" — {msg}" if msg else ""
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got} (want {want}){tail}",
              flush=True)
        if not ok:
            fails.append(name)

    task = compare_weather_task()
    print("== perfect multi_tool episode")
    tc = TOOL_CALL
    turns = [f'{tc}{{"name": "get_weather", "arguments": {{"city": "Tokyo"}}}}',
             f'{tc}{{"name": "get_weather", "arguments": {{"city": "New York"}}}}',
             "New York is warmer than Tokyo by 3.7 degrees (22.1C vs 18.4C)."]
    res = [{"city": "Tokyo", "temp": "18.4C"}, {"city": "New York", "temp": "22.1C"}]
    r = score_episode(task, turns, results=res)
    print(" ", r)
    expect("perfect.total", round(r["total"], 4), 1.0)
    expect("perfect.grounded", r["parts"]["grounded"], 1.0)
    expect("perfect.correct", r["parts"]["correct"], 1.0)

    print("== calls right, answer drops one city (the 0/8 failure mode)")
    turns_bad = list(turns[:2]) + ["Tokyo is 18.4C and New York is 22.1C."]
    r2 = score_episode(task, turns_bad, results=res)
    print(" ", r2)
    expect("dropped.total_above_task_correct_only", round(r2["total"], 4),
           round(0.2 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0 + 0.2 * 0.0, 4))
    expect("dropped.grounded_still_1", r2["parts"]["grounded"], 1.0,
           "both facts are adjacent to their own city here")
    expect("dropped.correct", r2["parts"]["correct"], 0.0,
           "no comparison stated")

    print("== values named but unbound (names both cities, swaps the values)")
    turns_swap = list(turns[:2]) + [
        "Tokyo is 22.1 degrees and New York is 18.4 degrees, so Tokyo is warmer "
        "by 3.7 degrees."]
    r3 = score_episode(task, turns_swap, results=res)
    print(" ", r3)
    expect("swap.grounded_penalised", r3["parts"]["grounded"], 0.0)

    print("== invented tool name")
    turns_inv = [f'{tc}{{"name": "get_live_temp", "arguments": {{"city": "Tokyo"}}}}',
                 "Tokyo is 18.4C."]
    r4 = score_episode(task, turns_inv,
                       results=[{"city": "Tokyo", "temp": "18.4C"}])
    print(" ", r4)
    expect("invented.declared", r4["parts"]["declared"], 0.0)
    expect("invented.correct", r4["parts"]["correct"], 0.0)

    print("== wrong arithmetic stated confidently")
    turns_wrong = list(turns[:2]) + [
        "New York is warmer than Tokyo by 40.7 degrees (22.1C vs 18.4C)."]
    r5 = score_episode(task, turns_wrong, results=res)
    print(" ", r5)
    expect("wrong_arith.correct", r5["parts"]["correct"], 0.0)

    print("== run to the token cap, two call blocks in one turn")
    turns_fmt = [f'{tc}{{"name": "get_weather", "arguments": {{"city": "Tokyo"}}}}\n'
                 f'{tc}{{"name": "get_weather", "arguments": {{"city": "New York"}}}}',
                 "New York is warmer by 3.7 degrees."]
    r6 = score_episode(task, turns_fmt, stopped=[False, True], results=res)
    print(" ", r6)
    expect("format.multi_block_penalised", r6["parts"]["format"], 0.7,
           "turn1 = 1 - 0.4 (no im_end) - 0.2 (two blocks), turn2 = 1, mean 0.7")

    print("== error recovery: report the failure, do not invent a temperature")
    er = error_recovery_task()
    r7 = score_episode(er, [f'{tc}{{"name": "get_weather", "arguments": {{"city": "Berlin"}}}}',
                               "Berlin is not in the database, so I can't report its weather."],
                       results=[{"error": "unknown city 'Berlin'. Valid: ['Tokyo', 'Paris', 'London', 'New York']"}])
    print(" ", r7)
    expect("recovery.correct", r7["parts"]["correct"], 1.0)
    expect("recovery.total", round(r7["total"], 4),
           round(0.2 * 1.0 + 0.3 * 1.0 + 0.3 * 0.0 + 0.2 * 1.0, 4))

    print("== do-nothing control (the AWM lesson: 1/12 tasks passed with zero calls)")
    for t in TASKS:
        r = score_episode(t, ["I don't know."], results=[])
        print(f"  {t.name}: {round(r['total'], 4)}")
        expect(f"donothing.{t.name}", r["total"], 0.0)

    print("== scripted two-turn recovery through the env loop")
    pol = _policy_script([
        (f'{tc}{{"name": "get_weather", "arguments": {{"city": "Berlin"}}}}', True),
        ("Berlin is not in the database, so I can't report its weather.", True),
    ])
    r8, ep = run_episode(er, pol, faults=["unknown_city"])
    print(" ", r8)
    expect("loop.recovery_correct", r8["parts"]["correct"], 1.0)

    print()
    if fails:
        print(f"SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    raise SystemExit(selftest() if args.selftest else 0)