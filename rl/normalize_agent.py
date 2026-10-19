#!/usr/bin/env python3
"""Normalize third-party agent trajectories onto TinyBalls' tool surface.

WHY THIS FILE EXISTS
--------------------
The surveyed trajectory datasets do NOT share our tool surface. Ours is exactly
`bash` / `str_replace_editor` / `submit`, rendered as
`<|tool_call|>{"name": ..., "arguments": {...}}` (RL_PLAN §3.1). Verified against
real first-rows on 2026-10-15:

  source                                    tool names seen                          verdict
  ----------------------------------------  ---------------------------------------  ---------------------
  SWE-bench/SWE-smith-trajectories (tool)   bash, str_replace_editor, submit         NATIVE (identical)
  SWE-Gym/OpenHands-SFT-Trajectories        execute_bash, str_replace_editor, finish RENAME-ONLY
  nebius/SWE-rebench-openhands-trajectories execute_bash, str_replace_editor, finish RENAME-ONLY
  SWE-Factory/DeepSWE-...Kimi-K2-2.8K       execute_bash, file_editor, finish        PROVISIONAL
  nvidia/Nemotron-RL-Agentic-Terminal       bash_command (free-JSON keystrokes)      OFF-SURFACE
  nebius/SWE-agent-trajectories             open/goto/edit (custom command language) OFF-SURFACE

`execute_bash -> bash` and `finish -> submit` are pure renames: the observed
argument schema is identical (`execute_bash{command}`, `finish{}`). `file_editor`
is NOT a rename of `str_replace_editor` — its observed edit args (command, path,
view_range, file_text) omit `old_str`/`new_str`, so a str_replace call there
would be fabricated, not translated. DeepSWE is therefore PROVISIONAL: it is
parsed but any `file_editor` call raises `Unmappable` and the row is dropped,
loudly. Nemotron's "tool" is a batch of timed keystrokes and SWE-agent's is a
custom `open`/`goto`/`edit` language; training our `<|tool_call|>` behavior on
either would teach the model tokens the env never accepts. They are OFF-SURFACE
and are kept as *task* sources only, never as behavior pairs.

The rule this module enforces: **never invent a tool call.** Every emitted call
is one the source model actually made, with its real arguments, only renamed when
the argument schema is provably identical.

Run:
    python rl/normalize_agent.py --audit
    python rl/normalize_agent.py --selftest
"""
from __future__ import annotations

import argparse
import json
import re

# ---------------------------------------------------------------------------
# our surface
# ---------------------------------------------------------------------------
NATIVE = ("bash", "str_replace_editor", "submit")

ALLOWED_ARGS = {
    "bash": {"command"},
    "str_replace_editor": {"command", "path", "view_range", "file_text",
                           "old_str", "new_str", "insert_line"},
    "submit": set(),
}
_INT_ARGS = {"insert_line"}

EDITOR_COMMANDS = ("view", "create", "str_replace", "insert")


def call(name: str, arguments: dict) -> tuple[str, dict]:
    """A canonical (name, arguments) pair, validated. Raises Unmappable."""
    if name not in NATIVE:
        raise Unmappable(f"tool {name!r} is not on our surface")
    if not isinstance(arguments, dict):
        raise Unmappable(f"{name}: arguments is {type(arguments).__name__}, not object")
    extra = set(arguments) - ALLOWED_ARGS[name]
    if extra:
        # An argument our tool has no slot for means the source tool differs from
        # ours in a way a rename cannot repair.
        raise Unmappable(f"{name}: unexpected argument(s) {sorted(extra)}")
    for k in list(arguments):
        if k in _INT_ARGS and not isinstance(arguments[k], int):
            try:
                arguments[k] = int(arguments[k])
            except (TypeError, ValueError):
                raise Unmappable(f"{name}: {k} is not an int: {arguments[k]!r}")
    if name == "str_replace_editor":
        cmd = arguments.get("command")
        if cmd not in EDITOR_COMMANDS:
            raise Unmappable(f"str_replace_editor: command {cmd!r} unsupported")
        if "path" not in arguments:
            raise Unmappable("str_replace_editor: no path")
        # Required args are per command. Without this a source row whose
        # `<parameter=new_str>` was dropped/empty is normalised into a valid
        # `str_replace` with no `new_str` — the exact call the env then deletes
        # text with (and round 5 paid for).
        if cmd == "create":
            if "file_text" not in arguments:
                raise Unmappable("str_replace_editor create: no file_text")
        elif cmd == "str_replace":
            for k in ("old_str", "new_str"):
                if k not in arguments or not str(arguments.get(k) or "").strip():
                    raise Unmappable(f"str_replace_editor str_replace: no {k}")
        elif cmd == "insert":
            if "new_str" not in arguments or not str(arguments.get("new_str") or "").strip():
                raise Unmappable("str_replace_editor insert: no new_str")
            if "insert_line" not in arguments:
                raise Unmappable("str_replace_editor insert: no insert_line")
    if name == "bash" and not str(arguments.get("command", "")).strip():
        raise Unmappable("bash: empty command")
    if name == "submit":
        arguments = {}
    return name, arguments


class Unmappable(Exception):
    pass


# ---------------------------------------------------------------------------
# source registry
# ---------------------------------------------------------------------------
SOURCES = {
    "swe-smith-trajectories": {
        "hf": "SWE-bench/SWE-smith-trajectories", "config": "default",
        "splits": ["tool", "xml", "ticks"],
        "fmt": "openai_tool_calls", "tool_map": {}, "obs_role": "tool",
        "surface": "native",
        "note": "bash/str_replace_editor/submit; system prompt byte-identical; "
                "observations role=tool"},
    "swe-gym-openhands": {
        "hf": "SWE-Gym/OpenHands-SFT-Trajectories", "config": "default",
        "splits": ["train.success.oss"],
        "fmt": "xml_function",
        "tool_map": {"execute_bash": "bash", "str_replace_editor": "str_replace_editor",
                     "finish": "submit"},
        "obs_role": "user", "surface": "renamed",
        "note": "XML <function=>; observations returned as role=user -> remapped"},
    "nebius-rebench-openhands": {
        "hf": "nebius/SWE-rebench-openhands-trajectories", "config": "default",
        "splits": ["train"],
        "fmt": "xml_function",
        "tool_map": {"execute_bash": "bash", "str_replace_editor": "str_replace_editor",
                     "finish": "submit"},
        "obs_role": "user", "surface": "renamed",
        "note": "same OpenHands surface; ships a `tools` json field + `resolved`"},
    "deepswe-kimi": {
        "hf": "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K", "config": "default",
        "splits": ["train"],
        "fmt": "xml_function",
        "tool_map": {"execute_bash": "bash", "finish": "submit"},
        "obs_role": "user", "surface": "provisional",
        "note": "file_editor is NOT str_replace_editor (missing old_str/new_str) -> dropped"},
    "nemotron-terminal": {
        "hf": "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1", "config": "default",
        "splits": ["train"],
        "fmt": "none", "tool_map": {}, "obs_role": None, "surface": "off",
        "note": "free-JSON {analysis,plan,commands[keystrokes]}; not tool calls"},
    "swe-agent": {
        "hf": "nebius/SWE-agent-trajectories", "config": "default",
        "splits": ["train"],
        "fmt": "none", "tool_map": {}, "obs_role": None, "surface": "off",
        "note": "custom open/goto/edit/... command language; not our tools"},
}


# ---------------------------------------------------------------------------
# content / call parsing
# ---------------------------------------------------------------------------
def as_text(content) -> str:
    """HF chat content is either a string or a list of {type,text} parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p)
                       for p in content)
    return str(content)


_FUNC_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_]*)>([\s\S]*?)</function>")
_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>([\s\S]*?)</parameter>")


def parse_xml_function(content: str):
    """OpenHands / DeepSWE: `<function=NAME><parameter=K>V</parameter></function>`.
    Returns (prose_without_calls, [(name, args_dict), ...])."""
    calls = []
    for fm in _FUNC_RE.finditer(content):
        args = {}
        for pm in _PARAM_RE.finditer(fm.group(2)):
            args[pm.group(1)] = _coerce(pm.group(1), pm.group(2).strip())
        calls.append((fm.group(1), args))
    prose = _FUNC_RE.sub("", content).strip()
    return prose, calls


def _coerce(key, val):
    if key == "view_range":
        try:
            v = json.loads(val)
            if isinstance(v, list):
                return [int(x) for x in v]
        except Exception:  # noqa: BLE001
            pass
        parts = re.split(r"[,\s]+", val.strip())
        try:
            return [int(x) for x in parts if x != ""]
        except ValueError:
            return val
    if key == "insert_line":
        try:
            return int(val)
        except ValueError:
            return val
    return val


def _openai_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            d = json.loads(raw)
            return d if isinstance(d, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


# ---------------------------------------------------------------------------
# the normalizer
# ---------------------------------------------------------------------------
def normalize(source: str, row: dict) -> dict:
    """One raw dataset row -> canonical turns. Raises Unmappable.

    Returns {"turns": [...], "workdir": str|None, "source_system": str}.
    `turns` are role system/user/assistant/tool dicts in the shape
    `sft/rollout.render_prefix` consumes: assistant carries `content`,
    optional `reasoning`, and `tool_calls=[{name,arguments}]`; tool carries
    `tool_name` + `content` (the payload, no marker).
    """
    spec = SOURCES[source]
    if spec["surface"] == "off":
        raise Unmappable(f"{source} is off-surface ({spec['note']})")

    raw = row.get("messages", row.get("trajectory"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise Unmappable(f"messages is a non-JSON string: {e}")
    if not isinstance(raw, list):
        raise Unmappable("no message list")

    tmap = spec["tool_map"]
    turns, system, waiting_obs = [], "", False
    for m in raw:
        role = m.get("role")
        if role == "system":
            system = as_text(m.get("content", m.get("system_prompt", "")))
            continue
        if role == "assistant":
            if spec["fmt"] == "xml_function":
                prose, raw_calls = parse_xml_function(as_text(m.get("content")))
            else:  # openai_tool_calls: {name, arguments-as-json-string}
                prose = as_text(m.get("content") or m.get("thought"))
                raw_calls = [(tc.get("function", {}).get("name") or tc.get("name"),
                              _openai_args(tc.get("function", {}).get("arguments")))
                             for tc in (m.get("tool_calls") or [])]
            canon = [call(tmap.get(n, n), dict(a)) for n, a in raw_calls]
            thought = as_text(m.get("thought"))
            turns.append({"role": "assistant", "content": prose,
                          "reasoning": thought if thought and thought != prose else "",
                          "tool_calls": [{"name": n, "arguments": a} for n, a in canon]})
            waiting_obs = bool(canon)
            continue
        if role in ("user", "tool"):
            text = as_text(m.get("content"))
            if waiting_obs and role == spec["obs_role"]:
                text = _strip_obs_wrapper(text)   # source header -> bare payload
                calls = turns[-1].get("tool_calls") if turns else None
                name = calls[0]["name"] if calls else "bash"
                turns.append({"role": "tool", "tool_name": name, "content": text})
                waiting_obs = False
            else:
                turns.append({"role": "user", "content": text})
            continue
        raise Unmappable(f"unexpected role {role!r}")

    if not any(t["role"] == "assistant" for t in turns):
        raise Unmappable("no assistant turns")
    workdir = _guess_workdir(turns)
    return {"turns": turns, "workdir": workdir, "source_system": system}


_OBS_HDR = re.compile(r"^Exit code:.*?\nExecution output of \[[^\]]+\]:\n", re.S)


def _strip_obs_wrapper(text: str) -> str:
    """Remove the source's observation header so the payload matches our
    `<|tool_result|>` payload convention (bare result). Sources observed:
    SWE-smith `OBSERVATION:\\n...`, DeepSWE `Exit code: N\\nExecution output of
    [tool]:\\n[STDOUT]\\n...`, OpenHands `Here's the result:\\nOBSERVATION:\\n...`."""
    i = text.find("OBSERVATION:")
    if i != -1:
        return text[i + len("OBSERVATION:"):].lstrip("\n")
    m = _OBS_HDR.match(text)
    if m:
        return text[m.end():]
    return text


def _guess_workdir(turns):
    """The repo root the source worked in (`/testbed`, `/workspace/...`), so the
    prompt can name a real path the same way the source did."""
    for t in turns:
        for blob in (t.get("content", ""), t.get("content", "") if isinstance(t.get("content"), str) else ""):
            if not isinstance(blob, str):
                continue
            for pat in (r"/testbed", r"/workspace/[A-Za-z0-9_.\-]+"):
                m = re.search(pat, blob)
                if m:
                    return m.group(0)
    return None


# ---------------------------------------------------------------------------
# audit + selftest
# ---------------------------------------------------------------------------
def audit() -> int:
    print(f"{'source':<26} {'surface':<12} tools")
    print("-" * 78)
    for name, s in SOURCES.items():
        tools = ", ".join(f"{k}->{v}" if k != v else k for k, v in
                          (s["tool_map"] or {t: t for t in ()}).items()) or "(off-surface)"
        print(f"{name:<26} {s['surface']:<12} {tools or s['note'][:44]}")
    print("\nNATIVE = use as-is. RENAME-ONLY = names differ, args identical.")
    print("PROVISIONAL = some tool unmappable -> rows dropped. OFF = never as pairs.")
    return 0


_FIX = {
    "openai_asst": json.dumps({
        "role": "assistant",
        "content": "Now, let's examine the content of this file.",
        "thought": "Now, let's examine the content of this file.",
        "tool_calls": [{"index": 1, "type": "function", "id": "toolu_x",
                        "function": {"name": "str_replace_editor",
                                     "arguments": "{\"command\": \"view\", \"path\": \"/testbed/djmoney/forms/widgets.py\"}"}}]}),
    "openai_tool": json.dumps({
        "role": "tool", "message_type": "observation",
        "content": [{"type": "text", "text": "OBSERVATION:\n/testbed/djmoney/forms/widgets.py"}]}),
    "xml_asst": "<function=str_replace_editor>\n<parameter=command>view</parameter>\n"
                "<parameter=path>/workspace/python__mypy__0.820</parameter>\n</function>",
    "xml_bash": "let me look\n<function=execute_bash>\n  <parameter=command>find /testbed -name \"*.py\" | head</parameter>\n</function>",
    "xml_finish": "<function=finish>\n</function>",
    "xml_file_editor": "<function=file_editor>\n<parameter=command>str_replace</parameter>\n"
                       "<parameter=path>/a.py</parameter>\n<parameter>zz</parameter>\n</function>",
}


def selftest() -> int:
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== registry sanity")
    ok("has_native", [s for s in SOURCES.values() if s["surface"] == "native"],
       "exactly SWE-smith should be native")
    ok("deepswe_provisional", SOURCES["deepswe-kimi"]["surface"] == "provisional")
    ok("nemotron_off", SOURCES["nemotron-terminal"]["surface"] == "off")
    ok("sweagent_off", SOURCES["swe-agent"]["surface"] == "off")

    print("== openai tool_calls (SWE-smith native)")
    row = {"messages": json.dumps([
        {"role": "system", "content": "You are a helpful assistant that can interact with a computer to solve tasks."},
        {"role": "user", "content": [{"type": "text", "text": "<uploaded_files>\n/testbed\n</uploaded_files>"}]},
        json.loads(_FIX["openai_asst"]),
        json.loads(_FIX["openai_tool"]),
    ])}
    n = normalize("swe-smith-trajectories", row)
    ok("native_roles", [t["role"] for t in n["turns"]] == ["user", "assistant", "tool"],
       str([t["role"] for t in n["turns"]]))
    tc = n["turns"][1]["tool_calls"][0]
    ok("native_call", tc["name"] == "str_replace_editor" and tc["arguments"]["path"].endswith("widgets.py"))
    ok("native_obs_stripped", n["turns"][2]["content"].startswith("/testbed/djmoney"),
       n["turns"][2]["content"][:40])
    ok("native_tool_name", n["turns"][2]["tool_name"] == "str_replace_editor")
    ok("native_workdir", n["workdir"] == "/testbed", str(n["workdir"]))

    print("== xml (OpenHands) — rename + observation remap + prose split")
    row = {"messages": [
        {"role": "system", "content": "You are a helpful assistant that can interact with a computer to solve tasks.\n<IMPORTANT>..."},
        {"role": "user", "content": "fix the bug in /workspace/python__mypy__0.820"},
        {"role": "assistant", "content": _FIX["xml_asst"]},
        {"role": "user", "content": "Here's the result:\nOBSERVATION:\nfile contents..."},
    ]}
    n = normalize("swe-gym-openhands", row)
    ok("xml_roles", [t["role"] for t in n["turns"]] == ["user", "assistant", "tool"],
       str([t["role"] for t in n["turns"]]))
    ok("xml_call", n["turns"][1]["tool_calls"][0] ==
       {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/workspace/python__mypy__0.820"}})
    ok("xml_prose", n["turns"][1]["content"] == "")
    ok("xml_obs_stripped", "OBSERVATION:" not in n["turns"][2]["content"],
       n["turns"][2]["content"][:40])

    print("== xml rename execute_bash->bash, finish->submit")
    n2 = normalize("swe-gym-openhands", {"messages": [
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": _FIX["xml_bash"]},
        {"role": "user", "content": "Exit code: 0\nExecution output of [execute_bash]:\n[STDOUT]\nfoo.py"}]})
    ok("xml_bash_name", n2["turns"][1]["tool_calls"][0]["name"] == "bash")
    ok("xml_bash_prose", n2["turns"][1]["content"] == "let me look",
       n2["turns"][1]["content"])
    n3 = normalize("swe-gym-openhands", {"messages": [
        {"role": "user", "content": "t"}, {"role": "assistant", "content": _FIX["xml_finish"]}]})
    ok("xml_finish_submit", n3["turns"][1]["tool_calls"][0] == {"name": "submit", "arguments": {}})

    print("== off-surface and unmappable are refused, never fabricated")
    for s in ("nemotron-terminal", "swe-agent"):
        try:
            normalize(s, {"messages": []})
            ok(f"{s}_refused", False, "did not raise")
        except Unmappable:
            ok(f"{s}_refused", True)
    try:
        normalize("deepswe-kimi", {"messages": [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": _FIX["xml_file_editor"]}]})
        ok("file_editor_dropped", False, "did not raise")
    except Unmappable:
        ok("file_editor_dropped", True)

    print("== argument validation")
    for bad, why in [
        ({"name": "bash", "arguments": {"cmd": "ls"}}, "unknown arg"),
        ({"name": "execute_bash", "arguments": {"command": "ls"}}, "off-surface tool"),
        ({"name": "str_replace_editor", "arguments": {"command": "append", "path": "a"}}, "bad editor cmd"),
        ({"name": "str_replace_editor", "arguments": {"command": "str_replace",
                                                      "path": "a", "old_str": "x"}}, "missing new_str"),
        ({"name": "str_replace_editor", "arguments": {"command": "insert",
                                                      "path": "a"}}, "missing insert new_str"),
        ({"name": "str_replace_editor", "arguments": {"command": "create",
                                                      "path": "a"}}, "missing file_text"),
        ({"name": "bash", "arguments": {"command": "  "}}, "empty command"),
    ]:
        try:
            call(bad["name"], dict(bad["arguments"]))
            ok(f"reject_{why.replace(' ', '_')}", False, "accepted")
        except Unmappable:
            ok(f"reject_{why.replace(' ', '_')}", True)

    print()
    if fails:
        print(f"NORMALIZE SELFTEST FAIL: {fails}")
        return 1
    print("NORMALIZE SELFTEST PASS")
    return 0


def probe(limit: int = 5) -> int:
    """Normalize real first-rows from the Hub — the check that the mapping holds
    on the actual data, not on fixtures. Off-surface sources are skipped."""
    import collections
    import urllib.parse
    import urllib.request

    rc = 0
    for name, s in SOURCES.items():
        if s["surface"] == "off":
            print(f"{name:<26} SKIP  off-surface ({s['note'][:40]})")
            continue
        url = ("https://datasets-server.huggingface.co/first-rows?dataset="
               + urllib.parse.quote(s["hf"])
               + f"&config={s['config']}&split={s['splits'][0]}")
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                rows = [x["row"] for x in json.load(r).get("rows", [])]
        except Exception as e:  # noqa: BLE001
            print(f"{name:<26} ERROR fetching: {e}")
            rc = 1
            continue
        good, tools = 0, collections.Counter()
        errs, trunc = collections.Counter(), 0
        for row in rows[:limit]:
            try:
                n = normalize(name, row)
                good += 1
                for t in n["turns"]:
                    for c in t.get("tool_calls") or []:
                        tools[c["name"]] += 1
            except Unmappable as e:
                msg = str(e)
                # datasets-server clips long string cells, so a big trajectory
                # arrives as an unterminated JSON string. That is a transport
                # limit, not a mapping failure; real ingestion reads via `datasets`.
                if "Unterminated string" in msg or "non-JSON" in msg:
                    trunc += 1
                else:
                    errs[msg[:50]] += 1
        print(f"{name:<26} {good}/{len(rows[:limit])} ok  tools={dict(tools)}"
              + (f"  truncated_by_api={trunc}" if trunc else "")
              + (f"  drops={dict(errs)}" if errs else ""))
        if good == 0 and trunc < len(rows[:limit]):
            rc = 1
    return rc



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--probe", type=int, nargs="?", const=5, default=None,
                    help="normalize N real first-rows per source from the Hub")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.probe is not None:
        raise SystemExit(probe(a.probe))
    raise SystemExit(audit())
