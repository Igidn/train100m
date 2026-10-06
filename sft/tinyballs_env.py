#!/usr/bin/env python3
"""Mock tool env for the Pi harness, with deterministic fault injection.

Normal path is identical to agent_test.py's run_tool. FaultState makes the
FIRST call of a given (tool, args) signature fail (so an agent must recover),
subsequent calls succeed. Enable in the harness with:

    TINYBALLS_FAULTS=unknown_city,wrong_key,transient,denied  python agent_test_tinyballs.py

Copied verbatim from the Pi (~/ai/sft-test/env.py, 2026-10-06) so the RL env, the
reward function and the eval harness all execute the same mock. Keep it that
way: the error payloads below are the same strings the error-recovery SFT rows
were generated from, so a rewording here is a training/serving mismatch. Changes
made on the Pi have to land here too.
"""
import json

WEATHER = {
    "Tokyo": {"temp_c": 18.4, "cond": "cloudy", "wind_kmh": 12},
    "Paris": {"temp_c": 11.2, "cond": "rain", "wind_kmh": 24},
    "London": {"temp_c": 9.7, "cond": "overcast", "wind_kmh": 31},
    "New York": {"temp_c": 22.1, "cond": "sunny", "wind_kmh": 8},
}
PEOPLE = {
    "Albert Einstein": {"born": 1879, "height_cm": 175, "field": "physics"},
    "Marie Curie": {"born": 1867, "height_cm": 155, "field": "physics/chemistry"},
    "Ada Lovelace": {"born": 1815, "height_cm": 168, "field": "computing"},
}
AVAILABLE = ["get_weather", "lookup_person", "calculate"]

FAULT_MODES = ("unknown_city", "wrong_key", "bad_type", "transient", "denied")


def run_tool(name, args):
    """Normal behavior (same results as the original harness)."""
    try:
        if name == "get_weather":
            w = WEATHER.get(args.get("city"))
            if w is None:
                return {"error": f"unknown city '{args.get('city')}'. Valid: {list(WEATHER)}"}
            unit = args.get("unit", "celsius")
            t = w["temp_c"]
            return {"city": args["city"],
                    "temp": f"{t}C" if unit == "celsius" else f"{t*9/5+32:.1f}F",
                    "condition": w["cond"], "wind_kmh": w["wind_kmh"]}
        if name == "lookup_person":
            p = PEOPLE.get(args.get("name"))
            return p if p else {"error": f"person '{args.get('name')}' not found"}
        if name == "calculate":
            expr = args.get("expression", "")
            import re
            if not re.fullmatch(r"[0-9.+\-*/() ]+", expr):
                return {"error": "invalid expression"}
            return {"result": str(eval(expr))}
        return {"error": f"unknown tool {name}"}
    except Exception as e:
        return {"error": str(e)}


def fault_text(mode, tool, args):
    if mode == "unknown_city":
        return f"unknown city '{args.get('city')}'. Valid: {list(WEATHER)}"
    if mode == "wrong_key":
        key = next(iter(args)) if args else "?"
        valid = {"get_weather": "city, unit", "lookup_person": "name",
                 "calculate": "expression"}.get(tool, "?")
        return f"unknown parameter '{key}' for {tool}. Valid parameters: {valid}"
    if mode == "bad_type":
        key = next(iter(args)) if args else "?"
        return (f"invalid argument type for '{key}': expected string, "
                f"got {type(args.get(key)).__name__}")
    if mode == "transient":
        return "upstream error 503: service temporarily unavailable"
    if mode == "denied":
        return "permission denied: this dataset requires an API key"
    return "unknown error"


class FaultState:
    """First call of each signature returns a fault; retries succeed.

    A fault is only injected when its precondition actually holds for the
    arguments (e.g. unknown_city only for a city that is not in the DB), so the
    environment never lies about a valid call.
    """

    VALID = {"get_weather": {"city", "unit"}, "lookup_person": {"name"},
             "calculate": {"expression"}}
    RANK = {"unknown_city": 0, "wrong_key": 1, "bad_type": 2, "denied": 3,
            "transient": 4}

    def __init__(self, modes):
        self.modes = sorted([m for m in modes if m in FAULT_MODES],
                            key=lambda m: self.RANK[m])
        self.seen = set()
        self.count = 0

    def _precondition(self, mode, name, args):
        if mode == "unknown_city":
            return (name == "get_weather" and isinstance(args.get("city"), str)
                    and args["city"] not in WEATHER)
        if mode == "wrong_key":
            valid = self.VALID.get(name)
            return bool(valid) and any(k not in valid for k in args)
        if mode == "bad_type":
            return any(isinstance(v, (list, dict)) for v in args.values())
        if mode == "transient":
            return name in AVAILABLE
        if mode == "denied":
            return name == "lookup_person"
        return False

    def take(self, name, args):
        """Return an error dict (once per signature) or None to run normally."""
        if not self.modes:
            return None
        sig = (name, json.dumps(args, sort_keys=True))
        if sig in self.seen:
            return None
        self.seen.add(sig)
        for mode in self.modes:  # most specific first
            if self._precondition(mode, name, args):
                self.count += 1
                return {"error": fault_text(mode, name, args)}
        return None
