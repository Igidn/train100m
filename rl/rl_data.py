#!/usr/bin/env python3
"""Good-data acquisition for the agentic RL plan — see rl/RL_PLAN.md §6.

Two modes, both safe to run on the Pi under ~/ai/data-venv:

  survey            read-only. For each source, print splits, schema, row counts
                    and one real row via the datasets-server API. No download,
                    no HF token, no deps beyond the stdlib.

  fetch <name>      stream a source, apply the plan's filter, write normalized
                    jsonl (+ manifest). Needs `datasets`.

The point of this file is the SPEC, not the plumbing: SOURCES below is the exact
list of "good data for what we are actually tuning for" — agentic behavior with
bash/read/write/edit, not code authorship. Do not add a source without a filter
that the plan names (resolved==true, target==true, ...), or it stops being gold.

  python rl/rl_data.py survey
  python rl/rl_data.py fetch swe_smith_traj --good
  python rl/rl_data.py fetch swe_agent_traj --bad
  python rl/rl_data.py fetch all --good

Local mirror of the Pi layout: ~/ai/sft-pipeline/raw-agent/<name>/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request

HF_API = "https://huggingface.co/api/datasets/{ds}"
DS_SERVER = "https://datasets-server.huggingface.co"

# ---------------------------------------------------------------------------
# The spec. Every entry: repo id, splits, the field(s) the plan consumes, and
# the filter that turns it into chosen (good) or rejected (bad).
# ---------------------------------------------------------------------------
SOURCES = {
    # --- primary gold: mini-swe-agent traces, tools are bash/str_replace_editor/submit
    "swe_smith_traj": {
        "repo": "SWE-bench/SWE-smith-trajectories",
        "splits": ["tool"],           # xml/ticks = same trajs, other renderings
        "fields": ["messages", "instance_id", "resolved", "model", "traj_id", "patch"],
        "filter": "resolved",         # bool; --good -> True, --bad -> False
        "role": "gold",
        "note": "tools == bash / str_replace_editor (view|create|str_replace|insert) / submit",
    },
    # --- free chosen AND rejected: SWE-agent 'target' is solved/not
    "swe_agent_traj": {
        "repo": "nebius/SWE-agent-trajectories",
        "splits": ["train"],
        "fields": ["instance_id", "model_name", "target", "trajectory",
                   "exit_status", "generated_patch", "eval_logs"],
        "filter": "target",
        "role": "gold+bad",
        "note": "open/edit/bash editor interface; target==true is chosen, false is rejected",
    },
    # --- OpenHands traces with resolved 0/1
    "openhands_rebench": {
        "repo": "nebius/SWE-rebench-openhands-trajectories",
        "splits": ["train"],
        "fields": ["trajectory_id", "instance_id", "repo", "trajectory", "tools",
                   "model_patch", "exit_status", "resolved"],
        "filter": "resolved",
        "role": "gold+bad",
    },
    # --- successes only
    "openhands_sft": {
        "repo": "SWE-Gym/OpenHands-SFT-Trajectories",
        "splits": ["train.success.oss"],
        "fields": ["messages"],
        "filter": None,
        "role": "gold",
    },
    # --- small, high-quality Kimi-K2 traces (file_editor + bash)
    "deepswe_kimi": {
        "repo": "SWE-Factory/DeepSWE-Agent-Kimi-K2-Trajectories-2.8K",
        "splits": ["train"],
        "fields": ["messages"],
        "filter": None,
        "role": "gold",
    },
    # --- turn-level RL: every row is a state -> gold next bash action. No env.
    "nemotron_terminal": {
        "repo": "nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1",
        "splits": ["train"],
        "fields": ["uuid", "task_name", "tool_name", "responses_create_params",
                   "expected_answer", "metadata"],
        "filter": None,
        "role": "turn-level-gold",
        "note": "expected_answer = {analysis, plan, commands[]} for tool_name==bash_command",
    },
    # --- task bank for tier-0 micro-repos + Kaggle-side eval
    "swe_smith_tasks": {
        "repo": "SWE-bench/SWE-smith",
        "splits": ["train"],
        "fields": ["instance_id", "problem_statement", "patch", "FAIL_TO_PASS",
                   "PASS_TO_PASS", "repo", "image_name"],
        "filter": None,
        "role": "seed-tasks",
    },
    # --- abstention pairs (tool-calling, not agentic): keep for "should I call at all"
    "when2call_pref": {
        "repo": "nvidia/When2Call",
        "splits": ["train_pref"],
        "fields": ["chosen", "rejected"],
        "filter": None,
        "role": "pref",
    },
}


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "rl_data/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def survey(only: str | None) -> int:
    names = [only] if only else list(SOURCES)
    for name in names:
        if name not in SOURCES:
            print(f"unknown source: {name}", file=sys.stderr)
            return 2
        src = SOURCES[name]
        ds = src["repo"]
        print(f"\n{'=' * 72}\n{name}  ->  {ds}   [{src['role']}]")
        try:
            info = _get_json(HF_API.format(ds=urllib.parse.quote(ds, safe="/")))
        except Exception as e:  # noqa: BLE001
            print(f"  UNAVAILABLE: {type(e).__name__}: {str(e)[:160]}")
            continue
        print(f"  downloads={info.get('downloads')} likes={info.get('likes')} "
              f"gated={info.get('gated')}")
        try:
            splits = _get_json(f"{DS_SERVER}/splits?dataset={urllib.parse.quote(ds, safe='')}")
            for s in splits.get("splits", []):
                print(f"  split {s['split']:<24} cfg={s['config']}")
            if not splits.get("splits"):
                print("  (no splits from server)")
                continue
            s0 = splits["splits"][0]
            fr = _get_json(
                f"{DS_SERVER}/first-rows?dataset={urllib.parse.quote(ds, safe='')}"
                f"&config={urllib.parse.quote(s0['config'], safe='')}"
                f"&split={urllib.parse.quote(s0['split'], safe='')}&length=1")
            feats = [f["name"] for f in fr.get("features", [])]
            print(f"  features: {feats}")
            row = (fr.get("rows") or [{}])[0].get("row", {})
            for k in src["fields"]:
                if k in row:
                    v = row[k]
                    s = v if isinstance(v, str) else json.dumps(v, default=str)
                    print(f"    {k}: {s[:220]}")
        except Exception as e:  # noqa: BLE001
            print(f"  server error: {type(e).__name__}: {str(e)[:160]}")
        print(f"  filter: {src['filter']}   note: {src.get('note', '')}")
    return 0


def fetch(name: str, good: bool, out_root: str, limit: int, split: str | None) -> int:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        print("fetch needs `datasets` (pip install datasets). survey does not.",
              file=sys.stderr)
        return 2
    if name not in SOURCES:
        print(f"unknown source: {name}", file=sys.stderr)
        return 2
    src = SOURCES[name]
    ds, dsplit = src["repo"], (split or src["splits"][0])
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{'good' if good else 'bad'}.jsonl")

    print(f"streaming {ds} [{dsplit}] ...")
    stream = load_dataset(ds, split=dsplit, streaming=True)
    key = src["filter"]
    want = good
    n = kept = 0
    with open(out_path, "w") as fh:
        for row in stream:
            n += 1
            if key is not None:
                v = row.get(key)
                if isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "yes")
                if bool(v) != want:
                    continue
            rec = {k: row.get(k) for k in src["fields"]}
            rec["_source"] = ds
            rec["_role"] = src["role"]
            fh.write(json.dumps(rec, default=str) + "\n")
            kept += 1
            if limit and kept >= limit:
                break
    manifest = {"source": ds, "split": dsplit, "filter": key, "want": want,
                "scanned": n, "kept": kept, "out": out_path}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  scanned {n}, kept {kept} ({key}={want}) -> {out_path}")
    return 0 if kept else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("name", nargs="?", default=None)
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--good", action="store_true", help="keep filter==true")
    f.add_argument("--bad", action="store_true", help="keep filter==false")
    f.add_argument("--split", default=None)
    f.add_argument("--limit", type=int, default=0)
    f.add_argument("--out", default=os.path.expanduser("~/ai/sft-pipeline/raw-agent"))
    a = ap.parse_args()
    if a.cmd == "survey":
        return survey(a.name)
    names = list(SOURCES) if a.name == "all" else [a.name]
    good = a.good or not a.bad  # default to good when neither is given
    rc = 0
    for n in names:
        rc |= fetch(n, good, a.out, a.limit, a.split)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
