#!/usr/bin/env python3
"""MiMo-V2.6-RL prompts-only loader — RL_PLAN_2 §6.2.

Honest read of `XiaomiMiMo/MiMo-V2.6-RL-oss` (verified 2026-10-16 via the
datasets-server):

  * it is **prompt diversity**, never reward. The verifiers live inside
    per-row docker images (`extra_info.instance_json.docker_image`) that do not
    run on a Mac or on Kaggle's CPU-only env;
  * its 7,780 rows are real GitHub issues on large repos, so our 113M student's
    pass rate on them is ~0 -> zero-variance groups -> zero gradient. Adding them
    as *tasks* would poison the pool;
  * the `prompt` field is a chat list, usually a single `user` message that is
    the problem statement, and it is **mixed English/Chinese**.

So this tool does the only transferable thing: turn a row into a clean English
problem statement with the environment references stripped, ready to be re-targeted
to our sandbox later (§6.2/§11 step 8). It authors no checks and grades nothing.

    python rl/mimo_prompts.py survey
    python rl/mimo_prompts.py survey --config general
    python rl/mimo_prompts.py fetch --config code --limit 500 --out rl/mimo/code.jsonl
    python rl/mimo_prompts.py --selftest           # pure functions, no network
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HF_API = "https://huggingface.co/api/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss"
DS = "XiaomiMiMo/MiMo-V2.6-RL-oss"
DS_SERVER = "https://datasets-server.huggingface.co"
CONFIGS = ["code", "cyber", "general", "webdev", "music"]

# CJK + fullwidth ranges; a problem statement that is mostly CJK is not English.
CJK_RE = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]")
# environment references that must not survive into a re-targeted prompt (§6.2):
# the fixed workdirs MiMo uses and its per-row image ids.
ENV_SUBS = [
    (re.compile(r"/workspace/repo\b"), "<REPO>"),
    (re.compile(r"/testbed\b"), "<REPO>"),
    (re.compile(r"\bformat-[a-z0-9-]*task-\d+\b"), "<IMAGE>"),
    (re.compile(r"\b[a-z0-9_.-]+/[a-z0-9_.-]+:latest\b"), "<IMAGE>"),
]


def _get_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "mimo_prompts/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------- pure
def is_english(text: str, min_latin: int = 40, cjk_frac: float = 0.05) -> bool:
    """A cheap, explainable filter. `langdetect` is a dependency and a coin flip
    on short strings; a problem statement is either roman-script or it is not."""
    cjk = len(CJK_RE.findall(text))
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    if latin < min_latin:
        return False
    return cjk / (cjk + latin + 1) <= cjk_frac


def strip_env(text: str) -> str:
    for pat, rep in ENV_SUBS:
        text = pat.sub(rep, text)
    # collapse the whitespace the rose from \r\n-heavy GitHub templates
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def extract_prompt(row: dict) -> str:
    """The problem statement, from `prompt` (chat list / str) or `extra_info`."""
    p = row.get("prompt")
    if isinstance(p, str):
        try:
            p = json.loads(p)
        except Exception:  # noqa: BLE001
            return strip_env(p)
    if isinstance(p, list):
        parts = [m.get("content", "") for m in p
                 if isinstance(m, dict) and m.get("role") in ("user", "system")]
        if any(parts):
            return strip_env("\n".join(x for x in parts if x))
    ei = row.get("extra_info")
    if isinstance(ei, str):
        try:
            ei = json.loads(ei)
        except Exception:  # noqa: BLE001
            ei = None
    if isinstance(ei, dict):
        ij = ei.get("instance_json")
        if isinstance(ij, str):
            try:
                ij = json.loads(ij)
            except Exception:  # noqa: BLE001
                ij = None
        if isinstance(ij, dict) and ij.get("problem_statement"):
            return strip_env(str(ij["problem_statement"]))
    return ""


def normalize(row: dict, config: str) -> dict | None:
    """A MiMo row -> a prompt record, or None if it is not usable English."""
    text = extract_prompt(row)
    if not text or not is_english(text):
        return None
    ei = row.get("extra_info")
    if isinstance(ei, str):
        try:
            ei = json.loads(ei)
        except Exception:  # noqa: BLE001
            ei = {}
    ei = ei if isinstance(ei, dict) else {}
    return {
        "id": f"mimo-{config}-{ei.get('instance_id', ei.get('index', '?'))}",
        "config": config,
        "ability": row.get("ability"),
        "data_source": row.get("data_source"),
        "agent_name": row.get("agent_name"),
        "prompt": text,
        "chars": len(text),
    }


# --------------------------------------------------------------------------- io
def survey(configs) -> int:
    for cfg in configs:
        print(f"\n{'=' * 72}\n{cfg}")
        try:
            fr = _get_json(f"{DS_SERVER}/first-rows?dataset="
                           f"{urllib.parse.quote(DS, safe='')}"
                           f"&config={cfg}&split=train&length=2")
        except Exception as e:  # noqa: BLE001
            print(f"  UNAVAILABLE: {type(e).__name__}: {str(e)[:160]}")
            continue
        feats = [f["name"] for f in fr.get("features", [])]
        print(f"  features: {feats}")
        kept = 0
        for rr in fr.get("rows", []):
            row = rr["row"]
            rec = normalize(row, cfg)
            print(f"  -- index {row.get('extra_info', {}).get('index')} "
                  f"ability={row.get('ability')} agent={row.get('agent_name')} "
                  f"english={'yes' if rec else 'NO'}")
            if rec:
                kept += 1
                print(f"     {rec['chars']} chars: {rec['prompt'][:180]!r}")
        print(f"  english in sample: {kept}/{len(fr.get('rows', []))} "
              f"(filter is a heuristic — see is_english)")
    return 0


def fetch(config, limit, out, offset=0) -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        print("fetch needs `datasets` (pip install datasets). survey does not.",
              file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"streaming {DS} [{config}] ...")
    stream = load_dataset(DS, config, split="train", streaming=True)
    n = kept = 0
    with open(out, "w") as fh:
        for row in stream:
            n += 1
            if n <= offset:
                continue
            rec = normalize(row, config)
            if rec is None:
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    print(f"  scanned {n}, kept {kept} English prompts -> {out}")
    return 0 if kept else 3


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("survey")
    s.add_argument("--config", default=None, help="one config; default all")
    f = sub.add_parser("fetch")
    f.add_argument("--config", default="code")
    f.add_argument("--limit", type=int, default=500)
    f.add_argument("--offset", type=int, default=0)
    f.add_argument("--out", default="rl/mimo/code.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == "survey":
        return survey([a.config] if a.config else CONFIGS)
    if a.cmd == "fetch":
        return fetch(a.config, a.limit, a.out, a.offset)
    ap.print_help()
    return 2


# --------------------------------------------------------------------------- selftest
def selftest() -> int:
    fails = []

    def ok(name, cond, msg=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{(' — ' + msg) if msg else ''}")
        if not cond:
            fails.append(name)

    print("== is_english separates roman from CJK problem statements")
    ok("english_yes", is_english("I want the Role REST API to keep a handle on create " * 2))
    ok("chinese_no", not is_english("我现在在 Cortex 里上传 Alertmanager 配置时，"
                                    "receiver 的 HTTP 通知鉴权基本只能用 basic auth"))
    ok("short_no", not is_english("fix it"))
    ok("mixed_mostly_en", is_english(
        "The bucket client keeps retrying after the request context is canceled, "
        "which floods the logs and hides the real failure. It should stop as soon "
        "as the caller's context is done, and it should surface the last error to "
        "the user instead of silently backing off forever. 少量中文说明" * 2))

    print("== strip_env removes the image and fixed workdirs")
    t = strip_env("Run in /testbed, image format-code-task-001457:latest; "
                  "cwd /workspace/repo okay")
    ok("testbed", "/testbed" not in t and "<REPO>" in t, t)
    ok("workspace", "/workspace/repo" not in t, t)
    ok("image", "format-code-task-001457" not in t and "<IMAGE>" in t, t)

    print("== extract_prompt: chat list, json str, and extra_info fallback")
    row1 = {"prompt": [{"role": "user", "content": "Fix the widget in /testbed."}]}
    ok("chat_list", extract_prompt(row1) == "Fix the widget in <REPO>.", extract_prompt(row1))
    row2 = {"prompt": json.dumps([{"role": "user", "content": "From a JSON string."}])}
    ok("json_str", extract_prompt(row2) == "From a JSON string.", extract_prompt(row2))
    row3 = {"prompt": [], "extra_info": {"instance_json": json.dumps(
        {"problem_statement": "Nested fallback in /testbed."})}}
    ok("extra_info", extract_prompt(row3) == "Nested fallback in <REPO>.",
       extract_prompt(row3))

    print("== normalize drops non-English and keeps the rest")
    row = {"data_source": "opensource-code", "ability": "swe",
           "agent_name": "mimo_swe_agent",
           "prompt": [{"role": "user", "content": "A perfectly good English issue. " * 3}],
           "extra_info": {"index": 7, "instance_id": "format-code-task-000007"}}
    rec = normalize(row, "code")
    ok("normalize_kept", rec is not None and rec["id"] == "mimo-code-format-code-task-000007",
       str(rec))
    ok("normalize_no_reward", rec is not None and "reward_model" not in rec
       and "docker" not in json.dumps(rec))
    bad = dict(row, prompt=[{"role": "user", "content": "请修复这个问题。" * 5}])
    ok("normalize_drops_cjk", normalize(bad, "code") is None)

    print()
    if fails:
        print(f"MIMO SELFTEST FAIL: {len(fails)} — {fails}")
        return 1
    print("MIMO SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
