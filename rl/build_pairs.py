#!/usr/bin/env python3
"""Stage A pair builder — turn-level DPO pairs for agentic behavior (RL_PLAN §2A).

For every assistant turn in a **gold** trajectory, emit one pair:

    prompt_msgs = everything before that turn        (render_prefix -> ids)
    chosen      = the gold turn's completion text
    rejected    = the student's completion text for the *same* prompt (teacher-forced)

Both sides are appended to `render_prefix(prompt_msgs)` by the trainer
(`rl/dpo_verify.build_pair_tensors`), so each side must carry its own turn
header. Confirmed against the real tokenizer (2026-07-… this checkout):

    render_prefix(pre) + tok.encode("<|im_start|>assistant\\n" + body) + [IM_END_ID]
        == render_prefix(pre + [turn])            # body = [<think>r</think>]content[<|tool_call|>{...}]

Gold trajectories come from two places, both landing in the one canonical shape
that `sft/rollout.render_prefix` consumes:

  --source oracle   tier-0 micro-tasks solved by the deterministic oracle
                    (`agent_env` + `agent_rollout.oracle_policy`) — real sandbox
                    tool results, no dataset needed
  --source hub      normalized third-party gold (`rl/normalize_agent`), which is
                    the only place the tool-surface trap is handled (§6.5)

`rejected` is produced by querying the student (Ollama) on the gold prefix. That
is deliberately *teacher-forcing*: it isolates the one behavior under test (what
the policy does at this state) from compounding early errors, which is what makes
the pairs comparable at 100M. The student never sees the env here — for hub gold
we cannot run the real repos on the Mac — so no sandbox is required.

Ref logprobs are NOT computed here: `logp_ref_chosen/rejected` are filled on
Kaggle as step 0 of the DPO job (RL_PLAN §2A). Rows carry them as null.

    # no model, structural + byte self-test
    python rl/build_pairs.py --selftest
    # tier-0 gold pairs with the live student
    python rl/build_pairs.py --source oracle --tasks 8 --out pairs-oracle.jsonl
    # hub gold (needs the `datasets` lib; on the Mac use venv312)
    python rl/build_pairs.py --source hub --dataset swe-gym-openhands \
        --rows 200 --out pairs-openhands.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .agent_env import TOOL_CALL, generate_task, run_episode, system_prompt
    from .agent_rollout import OllamaPolicy, oracle_policy
    from .normalize_agent import SOURCES, Unmappable, normalize
except ImportError:  # run as a plain script
    from agent_env import TOOL_CALL, generate_task, run_episode, system_prompt
    from agent_rollout import OllamaPolicy, oracle_policy
    from normalize_agent import SOURCES, Unmappable, normalize

IM_START = "<" + "|im_start|>"
IM_END_ID = 2
HEADER = IM_START + "assistant\n"


# --------------------------------------------------------------------------- turn text
def _strip_calls(text: str):
    """Split an assistant text into (prose, [(name,args)]) by locating the JSON
    call objects and excising them. Used when a turn arrives as flat text (the
    env's history) rather than as an explicit tool_calls list."""
    calls = []
    spans = []
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments", obj.get("parameters")), (dict, str))):
            args = obj.get("arguments", obj.get("parameters"))
            calls.append((obj["name"], args))
            start = j
            marker = text.rfind(TOOL_CALL, 0, j)   # swallow the <|tool_call|> marker
            if marker >= 0 and not text[marker + len(TOOL_CALL):j].strip():
                start = marker
            spans.append((start, j + end))
        i = j + end
    prose = text
    for a, b in reversed(spans):
        prose = prose[:a] + prose[b:]
    return prose.strip(), calls


def canonicalize(msg: dict) -> dict:
    """Any assistant message -> {content, reasoning, tool_calls}. Accepts both the
    structured form (`normalize_agent` output) and the env's flat history."""
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    if msg.get("tool_calls"):
        return {"content": content.strip(), "reasoning": reasoning,
                "tool_calls": [{"name": t["name"], "arguments": t["arguments"]}
                               for t in msg["tool_calls"]]}
    m = re.match(r"\s*<think>(.*?)</think>\s*", content, re.S)
    if m:
        reasoning = m.group(1).strip()
        content = content[m.end():]
    prose, calls = _strip_calls(content)
    return {"content": prose, "reasoning": reasoning,
            "tool_calls": [{"name": n, "arguments": a} for n, a in calls]}


def body(turn: dict) -> str:
    """The part of a turn after `<|im_start|>assistant\\n` (what the model emits)."""
    t = canonicalize(turn)
    out = ""
    if t["reasoning"]:
        out += f"<think>{t['reasoning']}</think>\n"
    if t["content"]:
        out += t["content"]
    for tc in t["tool_calls"]:
        out += TOOL_CALL + json.dumps({"name": tc["name"], "arguments": tc["arguments"]},
                                      ensure_ascii=False)
    return out


def completion(turn: dict) -> str:
    """What the trainer appends to the prompt for this turn (header included)."""
    return HEADER + body(turn)


def response_completion(resp: dict) -> tuple[str, bool]:
    """A student response -> (completion text, stopped). Reconstructs the trained
    call markers when Ollama parses the call natively and strips them from text."""
    text = (resp.get("text") or "").strip()
    calls = resp.get("calls") or []
    extra = ""
    for n, a in calls:
        if n and n not in text:
            extra += TOOL_CALL + json.dumps({"name": n, "arguments": a}, ensure_ascii=False)
    return HEADER + text + extra, bool(resp.get("stopped", True))


# --------------------------------------------------------------------------- query
def to_chat(msgs):
    """Canonical turns -> Ollama chat messages (assistant turns re-expanded with
    their `<|tool_call|>` markers so the model sees its own prior calls)."""
    out = []
    for m in msgs:
        r = m["role"]
        if r == "assistant":
            out.append({"role": "assistant", "content": body(m)})
        elif r == "tool":
            out.append({"role": "tool", "content": m.get("content", "")})
        else:
            out.append({"role": r, "content": m.get("content", "")})
    return out


def make_query(policy, retries=3, backoff=1.5):
    """`policy` behind a retry. A single transient Ollama 500/timeout must not
    abort a 1000-pair run (it did once, at pair 150)."""
    def query(prompt_msgs):
        msgs = to_chat(prompt_msgs)
        last = None
        for i in range(retries + 1):
            try:
                return policy(msgs)
            except Exception as e:  # noqa: BLE001 — urllib/http/timeout
                last = e
                if i < retries:
                    time.sleep(backoff * (i + 1))
        raise last
    return query


def scripted_query(text, calls=(), stopped=True):
    def query(prompt_msgs):
        return {"text": text, "calls": list(calls), "stopped": stopped}
    return query


# --------------------------------------------------------------------------- pairs
def pairs_from_trajectory(turns, query, tok, key, max_prompt_tok, max_turns):
    """One gold trajectory -> its turn-level pairs. `turns[0]` is the system."""
    pairs, drops = [], {}
    made = 0
    for i, m in enumerate(turns):
        if m.get("role") != "assistant":
            continue
        prompt_msgs = turns[:i]
        if tok is not None:
            n = len(render_prefix_ids(tok, prompt_msgs))
            if n > max_prompt_tok:
                drops["long_prompt"] = drops.get("long_prompt", 0) + 1
                continue
        chosen = completion(m)
        try:
            resp = query(prompt_msgs)
        except Exception as e:  # noqa: BLE001
            drops["query_error"] = drops.get("query_error", 0) + 1
            print(f"    ! query failed at {key}:{i}: {e}", flush=True)
            continue
        rejected, stopped = response_completion(resp)
        if not rejected.strip()[len(HEADER):].strip():
            drops["empty_rejected"] = drops.get("empty_rejected", 0) + 1
            continue
        if rejected.strip() == chosen.strip():
            drops["no_signal"] = drops.get("no_signal", 0) + 1
            continue
        pairs.append({
            "key": f"{key}:{i}",
            "source": key,
            "turn_index": i,
            "prompt_msgs": prompt_msgs,
            "chosen": chosen,
            "rejected": rejected,
            "rejected_stopped": stopped,
            "logp_ref_chosen": None,      # filled on Kaggle (RL_PLAN §2A)
            "logp_ref_rejected": None,
        })
        made += 1
        if made >= max_turns:
            break
    return pairs, drops


def render_prefix_ids(tok, msgs):
    from sft.rollout import render_prefix
    return render_prefix(tok, msgs)


def gold_oracle(seeds):
    for s in seeds:
        task = generate_task(s)
        ep = run_episode(task, oracle_policy(task))
        yield f"micro-{s}", ep["messages"]


def gold_hub(source, rows):
    """Stream a Hub split and normalize each row. Uses `datasets` (installed on
    the Mac venv312) because the datasets-server truncates long message strings."""
    from datasets import load_dataset
    spec = SOURCES[source]
    if spec["surface"] == "off":
        raise SystemExit(f"{source} is off-surface; refusing (normalize_agent §6.5)")
    ds = load_dataset(spec["hf"], split=spec["splits"][0], streaming=True)
    for n, row in enumerate(ds):
        if rows and n >= rows:
            break
        try:
            norm = normalize(source, row)
        except Unmappable:
            continue
        system = {"role": "system",
                  "content": system_prompt(norm["workdir"] or "/testbed")}
        yield f"{source}-{n}", [system] + norm["turns"]


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["oracle", "hub"], default="oracle")
    ap.add_argument("--tasks", type=int, default=8, help="oracle: #micro tasks")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--dataset", default="swe-gym-openhands", help="hub: SOURCES key")
    ap.add_argument("--rows", type=int, default=200, help="hub: max rows")
    ap.add_argument("--model", default=os.environ.get("AGENT_MODEL", "tinyballs"))
    ap.add_argument("--tokenizer", default=os.path.expanduser(
        "~/.cache/tinyballs-import/tokenizer.json"))
    ap.add_argument("--out", default="pairs.jsonl")
    ap.add_argument("--max-prompt-tok", type=int, default=2048)
    ap.add_argument("--max-turns", type=int, default=4, help="pairs per trajectory")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--append", action="store_true",
                    help="append to --out instead of truncating (for resume)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    tok = None
    if a.tokenizer and os.path.isfile(a.tokenizer):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        print(f"tokenizer: {a.tokenizer}")
    else:
        print("tokenizer: none (prompt-length filter disabled)")

    policy = OllamaPolicy(a.model)
    query = make_query(policy)

    if a.source == "oracle":
        trajs = gold_oracle(range(a.seed0, a.seed0 + a.tasks))
    else:
        trajs = gold_hub(a.dataset, a.rows)

    n_pairs, all_drops, traj_n = 0, {}, 0
    with open(a.out, "a" if a.append else "w") as fh:
        for key, turns in trajs:
            traj_n += 1
            pairs, drops = pairs_from_trajectory(
                turns, query, tok, key, a.max_prompt_tok, a.max_turns)
            for k, v in drops.items():
                all_drops[k] = all_drops.get(k, 0) + v
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
                n_pairs += 1
            fh.flush()
            print(f"  {key:<16} turns={len(turns):<3} pairs={len(pairs)} "
                  f"{drops if drops else ''}", flush=True)
            if a.max_pairs and n_pairs >= a.max_pairs:
                break
    print(f"\n{traj_n} trajectories -> {n_pairs} pairs in {a.out}; drops={all_drops}",
          flush=True)
    return 0 if n_pairs else 2


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== canonicalize: flat history and structured turns agree")
    flat = {"role": "assistant",
            "content": '<think>hmm</think>\nls first\n' + TOOL_CALL
                       + '{"name": "bash", "arguments": {"command": "ls"}}'}
    c = canonicalize(flat)
    ok("reasoning", c["reasoning"] == "hmm")
    ok("content", c["content"] == "ls first", repr(c["content"]))
    ok("call", c["tool_calls"] == [{"name": "bash", "arguments": {"command": "ls"}}])
    structured = {"role": "assistant", "content": "ls first", "reasoning": "hmm",
                  "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]}
    ok("agree", canonicalize(structured) == c)

    print("== completion carries the turn header; body does not")
    ok("header", completion(flat).startswith(HEADER))
    ok("body_no_header", not body(flat).startswith(HEADER))
    ok("body_has_call", TOOL_CALL in body(flat))

    print("== empty rejected is refused")
    turns = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
             {"role": "assistant", "content": "hi"}]
    pairs, drops = pairs_from_trajectory(turns, scripted_query(""), None, "t", 999, 4)
    ok("empty_refused", pairs == [] and "empty_rejected" in drops, str(drops))
    pairs, drops = pairs_from_trajectory(turns, scripted_query("hi"), None, "t", 999, 4)
    ok("identical_refused", pairs == [] and "no_signal" in drops, str(drops))

    print("== a good pair has the trainer's shape")
    pairs, _ = pairs_from_trajectory(turns, scripted_query("good answer"), None, "t", 999, 4)
    p = pairs[0]
    ok("keys", set(p) >= {"key", "prompt_msgs", "chosen", "rejected",
                          "rejected_stopped", "logp_ref_chosen"})
    ok("prompt_len", len(p["prompt_msgs"]) == 2)          # system + user
    ok("chosen_header", p["chosen"] == HEADER + "hi")
    ok("rejected_header", p["rejected"] == HEADER + "good answer")

    print("== BYTE INVARIANT: render_prefix(pre)+encode(chosen)+[im_end] == "
          "render_prefix(pre+[turn])")
    tokpath = os.path.expanduser("~/.cache/tinyballs-import/tokenizer.json")
    if os.path.isfile(tokpath):
        from tokenizers import Tokenizer
        from sft.rollout import render_prefix
        tok = Tokenizer.from_file(tokpath)
        cases = [
            {"role": "assistant", "content": "plain"},
            {"role": "assistant", "content": "", "reasoning": "why",
             "tool_calls": [{"name": "bash", "arguments": {"command": "ls -la"}}]},
            {"role": "assistant", "content": "done",
             "tool_calls": [{"name": "submit", "arguments": {}}]},
        ]
        for c2 in cases:
            pre = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "HELLO"}]
            full = render_prefix(tok, pre + [c2])
            seq = render_prefix(tok, pre) + list(tok.encode(completion(c2)).ids) + [IM_END_ID]
            ok(f"bytes:{c2.get('content','') or 'call'}", seq == full,
               f"{len(seq)} vs {len(full)}")
        # and a mid-conversation case (prompt ends with a tool turn)
        pre = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "HELLO"},
               {"role": "assistant", "content": "",
                "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]},
               {"role": "tool", "tool_name": "bash", "content": '{"exit_code": 0}'}]
        c3 = cases[1]
        full = render_prefix(tok, pre + [c3])
        seq = render_prefix(tok, pre) + list(tok.encode(completion(c3)).ids) + [IM_END_ID]
        ok("bytes:after_tool", seq == full, f"{len(seq)} vs {len(full)}")
    else:
        print("  SKIP (no tokenizer at ~/.cache/tinyballs-import/tokenizer.json)")

    print()
    if fails:
        print(f"BUILD_PAIRS SELFTEST FAIL: {fails}")
        return 1
    print("BUILD_PAIRS SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
