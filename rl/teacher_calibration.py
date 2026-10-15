#!/usr/bin/env python3
"""Teacher calibration gate for RLAIF — rl/RL_PLAN.md §7.1.

Stage B keeps a trajectory only if the teacher labels it good/bad. That is only
worth doing if the teacher can *actually separate* good from bad. So before any
rollout loop exists, hand the teacher known labels and measure it:

  good = SWE-bench/SWE-smith-trajectories resolved==true   (or SWE-agent target==true)
  bad  = the same datasets resolved==false / target==false

and report accuracy, AUC(P(good)>P(bad)), the confusion matrix, the score gap,
AND the throughput (seconds/verdict, prompt+eval tokens) — the Pi number the
plan marks unverified, because a 1B model on a Raspberry Pi may be too slow to
sit in the loop regardless of whether it is accurate.

Verdict thresholds (the plan's §7.1 gate):

  PASS      acc >= 0.70 and auc >= 0.75   -> Stage B is real, build the loop
  MARGINAL  0.60 <= acc < 0.70            -> usable only blended with the
                                             deterministic checks; weight down
  FAIL      acc < 0.60                    -> coin flip; move to a bigger teacher
                                             (the family ships a 35B-A3B GGUF)

Torch-free, HTTP only to Ollama. Reads the jsonl that `rl/rl_data.py fetch`
writes, or any {label, ...} jsonl.

Run on the Pi, student not required:
  ~/ptts/bin/python3 teacher_calibration.py --good good.jsonl --bad bad.jsonl --limit 50
  python rl/rl_data.py fetch swe_smith_traj --good   # then --bad
  python teacher_calibration.py --selftest            # no model, tests metrics
  python teacher_calibration.py --dry-run --good good.jsonl --limit 1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
TEACHER = os.environ.get("TEACHER_MODEL", "openbmb/minicpm-v4.7")

RUBRIC = (
    "You are grading the BEHAVIOR of an AI agent that was given a software task "
    "and a shell (bash) plus a file editor (read/write/edit).\n"
    "You do NOT grade whether its code is correct or its patch is good. You grade "
    "whether it behaved like a competent agent, specifically:\n"
    "  1. did it INSPECT files before editing them (read-before-write)?\n"
    "  2. did it use the tools sensibly for the step it was on?\n"
    "  3. did it REACT to command output (including errors) instead of ignoring it?\n"
    "  4. did it AVOID repeating an identical failing action?\n"
    "  5. did it TERMINATE (submitted / gave a final answer) rather than running on?\n"
    "A short, correct sequence is better than a long rambling one. Editing a file "
    "it never looked at is bad; reading many files and never acting is bad.\n"
    "Reply with ONLY a JSON object, no prose:\n"
    '{"verdict": "good" | "bad", "score": <float 0..1>, '
    '"reasons": ["...", "..."]}'
)


# ---------------------------------------------------------------------------
# record loading / rendering
# ---------------------------------------------------------------------------

def _load_json_maybe(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:  # noqa: BLE001
            return None
    return v


def _content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" or "text" in c:
                    parts.append(str(c.get("text", "")))
                elif c.get("type") == "tool_result":
                    parts.append(_content_text(c.get("content")))
            else:
                parts.append(str(c))
        return "\n".join(p for p in parts if p)
    return str(content)


def flatten_message(m: dict) -> tuple[str, str]:
    """One record message -> (role, text) with tool calls inlined."""
    role = m.get("role", "?")
    text = _content_text(m.get("content") or m.get("text") or "")
    calls = m.get("tool_calls") or []
    rendered = []
    for c in calls:
        fn = (c or {}).get("function", {})
        name = fn.get("name", "?")
        args = fn.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.dumps(json.loads(args), ensure_ascii=False)
            except Exception:  # noqa: BLE001
                pass
        else:
            args = json.dumps(args, ensure_ascii=False)
        rendered.append(f"[{name}({args})]")
    if rendered:
        text = (text + " " if text else "") + " ".join(rendered)
    return role, text


def turns_from_record(rec: dict) -> tuple[str, list[tuple[str, str]]]:
    """Best-effort across the source schemas -> (task_text, [(role, text)])."""
    msgs = _load_json_maybe(rec.get("messages"))
    if msgs is None:
        msgs = _load_json_maybe(rec.get("trajectory"))
    if msgs is None:
        msgs = rec.get("turns") or []
    turns = [flatten_message(m) for m in msgs if isinstance(m, dict)]

    task = ""
    for k in ("problem_statement", "task", "issue"):
        if rec.get(k):
            task = str(rec[k])
            break
    if not task:
        # swe-smith puts the PR text in the first user turn; swe-agent in system
        for role, _ in turns:
            if role == "user":
                break
        for role, text in turns:
            if role in ("user",) and text.strip():
                task = text
                break
    return task, turns


def truncate_middle(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    head = budget * 3 // 5
    tail = budget - head - 24
    return text[:head] + "\n... [truncated] ...\n" + text[-tail:]


def render_prompt(task: str, turns: list[tuple[str, str]], budget: int) -> str:
    body = "\n".join(f"{r.upper()}: {t.strip()[:1200]}" for r, t in turns)
    body = truncate_middle(body, budget)
    t = truncate_middle(task.strip(), 1200)
    return f"TASK:\n{t or '(see trajectory)'}\n\nTRAJECTORY:\n{body}\n\nGrade the behavior."


def infer_label(rec: dict) -> str | None:
    if rec.get("label") in ("good", "bad"):
        return rec["label"]
    for k in ("resolved", "target"):
        if k in rec and rec[k] is not None:
            v = rec[k]
            if isinstance(v, str):
                v = v.strip().lower() in ("1", "true", "yes")
            return "good" if bool(v) else "bad"
    return None


def load_records(paths: list[str], label: str | None, label_field: str) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                lab = label or rec.get(label_field) or infer_label(rec)
                if lab not in ("good", "bad"):
                    continue
                out.append({"label": lab, "rec": rec})
    return out


# ---------------------------------------------------------------------------
# the judge
# ---------------------------------------------------------------------------

def _json_objects(text: str):
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
        except Exception:  # noqa: BLE001
            i = j + 1


def judge(prompt: str, model: str, url: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": RUBRIC},
                     {"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "num_predict": 192, "num_ctx": 8192},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    dt = time.time() - t0
    content = (resp.get("message") or {}).get("content") or ""
    verdict = None
    for obj in _json_objects(content):
        if isinstance(obj, dict) and "verdict" in obj:
            verdict = obj
            break
    if verdict is None:
        verdict = {"verdict": "bad", "score": 0.0, "reasons": ["unparseable"]}
    verdict["_secs"] = dt
    verdict["_prompt_tok"] = resp.get("prompt_eval_count", 0)
    verdict["_eval_tok"] = resp.get("eval_count", 0)
    return verdict


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def report(rows: list[dict]) -> dict:
    good = [r for r in rows if r["label"] == "good"]
    bad = [r for r in rows if r["label"] == "bad"]
    tp = sum(1 for r in good if r["verdict"] == "good")
    tn = sum(1 for r in bad if r["verdict"] == "bad")
    n = len(rows) or 1
    acc = (tp + tn) / n
    g = [float(r["score"]) for r in good]
    b = [float(r["score"]) for r in bad]
    secs = [r["_secs"] for r in rows]
    out = {
        "n": len(rows), "n_good": len(good), "n_bad": len(bad),
        "accuracy": round(acc, 4),
        "auc": round(auc(g, b), 4),
        "confusion": {"good_pred_good": tp, "good_pred_bad": len(good) - tp,
                      "bad_pred_bad": tn, "bad_pred_good": len(bad) - tn},
        "score_mean_good": round(statistics.mean(g), 4) if g else None,
        "score_mean_bad": round(statistics.mean(b), 4) if b else None,
        "score_gap": round((statistics.mean(g) - statistics.mean(b)), 4)
        if g and b else None,
        "secs_per_verdict": round(statistics.mean(secs), 2) if secs else None,
        "prompt_tok_mean": round(statistics.mean([r["_prompt_tok"] for r in rows]), 1)
        if rows else None,
    }
    if acc >= 0.70 and out["auc"] >= 0.75:
        out["gate"] = "PASS"
    elif acc >= 0.60:
        out["gate"] = "MARGINAL"
    else:
        out["gate"] = "FAIL"
    return out


# ---------------------------------------------------------------------------

def selftest() -> int:
    # a perfect judge and a coin-flip judge; metrics must separate them
    perfect = ([{"label": "good", "verdict": "good", "score": 0.9, "_secs": 1,
                 "_prompt_tok": 10} for _ in range(10)] +
               [{"label": "bad", "verdict": "bad", "score": 0.1, "_secs": 1,
                 "_prompt_tok": 10} for _ in range(10)])
    r = report(perfect)
    assert r["accuracy"] == 1.0 and r["auc"] == 1.0 and r["gate"] == "PASS", r
    flip = ([{"label": "good", "verdict": "good", "score": 0.5, "_secs": 1, "_prompt_tok": 1}
             for _ in range(5)] +
            [{"label": "good", "verdict": "bad", "score": 0.5, "_secs": 1, "_prompt_tok": 1}
             for _ in range(5)] +
            [{"label": "bad", "verdict": "good", "score": 0.5, "_secs": 1, "_prompt_tok": 1}
             for _ in range(5)] +
            [{"label": "bad", "verdict": "bad", "score": 0.5, "_secs": 1, "_prompt_tok": 1}
             for _ in range(5)])
    r2 = report(flip)
    assert r2["accuracy"] == 0.5 and r2["gate"] == "FAIL", r2
    # renderer round-trip on an swe-smith-shaped record
    rec = {"resolved": True, "messages": json.dumps([
        {"role": "system", "content": "You can interact with a computer."},
        {"role": "user", "content": [{"type": "text", "text": "fix the bug in foo.py"}]},
        {"role": "assistant", "content": "let me look",
         "tool_calls": [{"function": {"name": "bash",
                                      "arguments": json.dumps({"command": "ls"})}}]},
        {"role": "tool", "content": [{"type": "text", "text": "foo.py bar.py"}]},
    ])}
    task, turns = turns_from_record(rec)
    assert "fix the bug" in task, task
    p = render_prompt(task, turns, 500)
    assert "bash" in p and "foo.py" in p and p.startswith("TASK:"), p
    assert infer_label(rec) == "good"
    assert truncate_middle("x" * 100, 20).count("truncated") == 1
    print("teacher_calibration selftest OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--good", nargs="*", default=[])
    ap.add_argument("--bad", nargs="*", default=[])
    ap.add_argument("--data", nargs="*", default=[], help="jsonl with a label field")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--url", default=OLLAMA)
    ap.add_argument("--limit", type=int, default=50, help="per class")
    ap.add_argument("--budget", type=int, default=6000, help="trajectory char budget")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="calibration-report.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    rng = random.Random(a.seed)
    recs = load_records(list(a.good), "good", a.label_field) + \
        load_records(list(a.bad), "bad", a.label_field) + \
        load_records(list(a.data), None, a.label_field)
    if not recs:
        print("no records: pass --good/--bad/--data (or --selftest)", file=sys.stderr)
        return 2

    goods = [r for r in recs if r["label"] == "good"][:a.limit]
    bads = [r for r in recs if r["label"] == "bad"][:a.limit]
    todo = goods + bads
    rng.shuffle(todo)
    print(f"calibrating {a.teacher}: {len(goods)} good + {len(bads)} bad")

    if a.dry_run:
        task, turns = turns_from_record(todo[0]["rec"])
        print(render_prompt(task, turns, a.budget))
        return 0

    rows = []
    for i, item in enumerate(todo, 1):
        task, turns = turns_from_record(item["rec"])
        prompt = render_prompt(task, turns, a.budget)
        try:
            v = judge(prompt, a.teacher, a.url, a.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] judge error: {type(e).__name__}: {str(e)[:120]}")
            continue
        rows.append({"label": item["label"], "verdict": v["verdict"],
                     "score": v.get("score", 0.0), "reasons": v.get("reasons", []),
                     "_secs": v["_secs"], "_prompt_tok": v["_prompt_tok"],
                     "_eval_tok": v["_eval_tok"]})
        print(f"  [{i}/{len(todo)}] label={item['label']:<4} "
              f"verdict={v['verdict']:<4} score={v.get('score')} "
              f"({v['_secs']:.1f}s) {v.get('reasons', [''])[0][:60]}")

    rep = report(rows)
    rep["teacher"] = a.teacher
    rep["budget_chars"] = a.budget
    with open(a.out, "w") as fh:
        json.dump({"report": rep, "rows": rows}, fh, indent=1)
    print("\n" + json.dumps(rep, indent=1))
    print(f"\ngate: {rep['gate']}   -> {a.out}")
    return 0 if rep["gate"] != "FAIL" else 3


if __name__ == "__main__":
    raise SystemExit(main())
