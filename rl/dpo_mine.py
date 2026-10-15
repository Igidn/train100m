#!/usr/bin/env python
"""Mine DPO candidate pairs from the SFT corpus — stage 1 of 3 (`rl/`).

The task is the one gate that fails: `multi_tool` is 0/8 task-correct through the
unified serving path while scoring 8/8 on the calls themselves
(`REPAIR2_EVAL.md` §3). So the entire remaining error is the *final answer's*
entity binding — the model names both entities and attaches the wrong value, or
drops one. That shapes everything below:

  teacher-forced, not a rollout. Keep the gold scaffold (system + user + the gold
  `assistant[tool_calls]` + the gold `tool` results) as the prompt and let V1
  generate **only the final answer**. One prefill, one decode, no env loop, no
  shadow env — the observations are already in the prompt because they came from
  gold. A pair built this way is trained on exactly the thing that is broken.

  gold is `chosen`, V1's answer is `rejected`. The known hazard is style
  transfer: gold comes from a much larger teacher and writes bulleted markdown
  where V1 writes a line, so most of the gradient can go to prose. `rl/dpo_pairs.py`
  measures that per pair (`pure_binding`) rather than assuming it either way.

Runs on the Pi, CPU only, no torch — the tokenizer is the only dependency. The
yield number this prints (fraction of rows whose *gold* answer binds every mined
claim) is the gate on whether the rest of the pipeline is worth running: if it is
low the miner is wrong, not the `build_multi_mix.classify` filter it has to agree
with.

    python rl/dpo_mine.py --in ~/ai/sft-pipeline/norm-multi/multi-dolci-0000.jsonl.zst \
        --out dpo-mined
    python rl/dpo_mine.py --selftest
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# reward.py imports tinyballs_env as a top-level module, so sft/ has to be on
# the path too, not just the repo root.
for _p in (os.path.join(_ROOT, "sft"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from sft.reward import (normalize_text, same_entity_set,  # noqa: E402
                        score_grounded_pairs, value_present)

# The harness scenarios, decontaminated at SFT time. A mined row is new text and
# the eval is re-measured after DPO, so the filter has to run again here: a
# rollout that happens to say "Tokyo ... 18.4" is a leak into the only score
# that counts. Copied from build_multi_mix.py rather than imported, because that
# module imports pyarrow and the Pi's copy may drift.
EVAL_PATTERNS = [
    re.compile(r"tokyo[^\n]{0,80}18\.4|18\.4[^\n]{0,80}tokyo", re.I),
    re.compile(r"new york[^\n]{0,80}22\.1|22\.1[^\n]{0,80}new york", re.I),
    re.compile(r"berlin[^\n]{0,60}weather|weather[^\n]{0,60}berlin", re.I),
    re.compile(r"\beinstein\b", re.I),
    re.compile(r"\bcurie\b", re.I),
    re.compile(r"\blovelace\b", re.I),
]

MIN_CALLS = 2      # "multi-toolcall" — one call is the single-tool track
MIN_CLAIMS = 2     # fewer than 2 claims cannot express a binding failure
MAX_CLAIMS = 12    # above this the answer is a table, not a binding exercise
MAX_PROMPT_TOK = 1536

# Sanity floor on `gold_yield`, the fraction of rows reaching the gold-binding
# check that pass it. Measured 0.81 on multi-dolci; the residual 3.6% were
# hand-inspected and are table-shaped answers (a header entity over a table, a
# year used as the anchor, dates as entities) where no single answer shape binds
# entity to value — correctly rejected, not a broken miner.
#
# The floor is set low on purpose: its job is to catch a *systematically* broken
# miner (wrong entity extraction lands near 0.1), not to enforce a yield. An
# earlier 0.90 was wrong because it assumed build_multi_mix.classify was a binding
# filter; it is a bag-of-facts check, so a strict gate can never agree with it.
GOLD_YIELD_FLOOR = 0.50


# ------------------------------------------------------------------ pairing
def positional_pairs(msgs):
    """[(name, args, result), ...] for a normalized conversation.

    `tool` messages carry only `role` and `content` — no `tool_name` — so
    call-to-result pairing is purely positional and has to be reconstructed by
    walking `pending`, the same way `build_multi_mix.convert` does. Getting this
    wrong mines every claim against the wrong result and does not raise: the
    output still looks like plausible (entity, value) pairs. Hence the explicit
    None returns rather than a best-effort zip.

    Results are parsed with parse_float/parse_int=str so a float survives as the
    literal the dataset printed ("0.3476", not 0.34760000000000001). The answer
    is graded by substring, so it has to match the printed form.
    """
    out, pending = [], []
    for m in msgs:
        if m["role"] == "assistant":
            for tc in (m.get("tool_calls") or []):
                try:
                    args = json.loads(tc.get("arguments") or "{}")
                except Exception:
                    return None
                if not isinstance(args, dict):
                    return None
                pending.append((tc.get("name"), args))
        elif m["role"] == "tool":
            if not pending:
                return None          # a result with no call
            name, args = pending.pop(0)
            try:
                res = json.loads(m["content"], parse_float=str, parse_int=str)
            except Exception:
                res = m["content"]
            out.append((name, args, res))
    return None if pending else out  # dangling calls


def flatten(o, prefix=()):
    """[(path, leaf)] over nested dicts/lists."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, prefix + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from flatten(v, prefix + (str(i),))
    else:
        yield prefix, o


def _scalar(v):
    """A leaf worth quoting: numbers and short strings, not keys or blobs."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and NUMERIC.match(v.strip())):
        s = str(v).strip()
        return s if s and s not in ("0",) else None
    return None


NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def entity_candidates(args):
    """Strings in `args` usable as the entity a result belongs to.

    A comma-joined argument ("80057281,80189685") is split, because the answer
    names each element separately — treating the join as one entity asks the
    answer to quote a string that appears in no prose.
    """
    out = []
    for v in args.values():
        if not isinstance(v, str):
            continue
        v = v.strip()
        if len(v) < 3:
            continue
        parts = [p.strip() for p in v.split(",")] if "," in v else [v]
        for p in parts:
            if len(p) >= 3:
                out.append(p)
    return out


def mine_claims(pairs):
    """[(entity, value)] the final answer must bind, from the gold trace.

    entity: a string argument that also appears as a string in the result
            (geo.geocode {"address": "Kampala"} -> {"city": "Kampala", ...}).
            Falls back to the longest string argument when the tool does not echo
            its key back, which is the common case for a renamed field.
    value : the numeric leaves of that result, which is what the answer has to
            attach to *its own* entity.

    A **list** result is split per element and each element is attributed only to
    the entity it names. Flattening the whole list gives every entity every
    element's values -- for `[{id: 80057281, year: 2016}, {id: 80189685,
    year: 2019}]` that invents claims like (80057281, 2019), which no correct
    answer can satisfy.

    This is the judgement call in the pipeline and it has no oracle, so
    `selftest` pins the cases and the caller checks the yield against gold.
    """
    claims, seen = [], set()
    # Identifiers that appear as argument strings anywhere in the row are
    # entities, not measurements: for get_market({"ids": "80057281,80189685"})
    # the value 80189685 is the *other* entity's name, and claiming
    # (80057281, 80189685) asks the answer to bind an id to an id.
    all_idents = {a.strip().lower() for _, args, _ in pairs
                  for a in entity_candidates(args)}

    def emit(entity, leaves):
        for _, v in leaves:
            val = _scalar(v)
            if val is None or val.lower() == entity.lower():
                continue
            if val.strip().lower() in all_idents:
                continue
            key = (entity.lower(), val)
            if key in seen:
                continue
            seen.add(key)
            claims.append((entity, val))

    for name, args, res in pairs:
        # One entity per call. This is the shape the eval actually measures:
        # multi_tool is two get_weather calls, one per city, and the failure is
        # binding each city to its own temperature. Allowing several candidate
        # entities per call lets table-shaped results in — conditions
        # ("USED_ACCEPTABLE"), years, dates — where no single answer shape binds
        # entity to value, and the gradient becomes style transfer into a table.
        cands = entity_candidates(args)
        if not cands:
            continue
        if isinstance(res, list):
            # multi-record results are table-shaped; per-element attribution is
            # guesswork once a record mentions more than one identifier
            if len(res) > 2:
                continue
            for el in res:
                leaves = list(flatten(el))
                res_strs = {str(v).strip().lower() for _, v in leaves
                            if isinstance(v, str)}
                ents = [e for e in cands
                        if e.lower() in res_strs
                        or any(e.lower() in s for s in res_strs)]
                for e in (ents[:1] or cands[:1]):
                    emit(e, leaves)
            continue

        leaves = list(flatten(res))
        res_strs = {str(v).strip().lower() for _, v in leaves if isinstance(v, str)}
        ents = [e for e in cands
                if e.lower() in res_strs or any(e.lower() in s for s in res_strs)]
        for e in (ents[:1] or cands[:1]):
            emit(e, leaves)
    return claims


def contaminated(text):
    return any(p.search(text) for p in EVAL_PATTERNS)


def dedup_key(msgs):
    h = hashlib.md5()
    for m in msgs:
        h.update((m["role"] + "\x00" + (m.get("content") or "")).encode())
        for tc in m.get("tool_calls") or []:
            h.update((tc.get("name", "") + json.dumps(tc.get("arguments"),
                                                      sort_keys=True)).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ row gate
def build_row(msgs, prompt_len_fn, min_claims, max_claims, max_prompt_tok):
    """A mined row, or (None, reason). Every gate that can silently produce a
    useless pair is here, in cheapest-first order."""
    if len(msgs) < 3 or msgs[-1]["role"] != "assistant":
        return None, "shape"
    final = msgs[-1].get("content") or ""
    if not final.strip() or msgs[-1].get("tool_calls"):
        return None, "bad_final"
    if not any(m["role"] == "assistant" and m.get("tool_calls") for m in msgs):
        return None, "no_calls"

    pp = positional_pairs(msgs)
    if pp is None:
        return None, "pairing"
    if len(pp) < MIN_CALLS:
        return None, "one_call"

    claims = mine_claims(pp)
    if len(claims) < min_claims:
        return None, "few_claims"

    # Drop claims the gold answer never quotes. Real answers elide ("… about 60
    # additional daily closes …", "volume: ≈ 123 k"), and demanding they all be
    # bound rejects correct rows — measured at a 23% yield on the real corpus
    # before this. A claim nobody quotes is not a binding exercise.
    #
    # What survives is still a *stricter* filter than build_multi_mix.classify:
    # classify only asks that the facts appear somewhere (a bag-of-facts check,
    # which score_grounded's docstring calls out as teaching nothing), while
    # this additionally demands each value sit next to its own entity.
    gnorm = normalize_text(final)
    quoted = [(e, v) for e, v in claims if value_present(gnorm, v)]
    if len(quoted) < min_claims:
        return None, "few_quoted_claims"

    # The entity must be a *unique* anchor in the answer. `score_grounded_pairs`
    # cuts each entity's span at the next mention of any entity, so a repeated
    # entity ("female" twice, once per health question) makes the span ambiguous
    # and the clause test reads a value that belongs to the other mention. A row
    # whose entity is mentioned twice is not a binding exercise.
    def mentions(text_norm, needle):
        if len(needle) < 2:
            return 0
        return len(re.findall(r"(?<![a-z0-9])" + re.escape(needle)
                              + r"(?![a-z0-9])", text_norm))

    anchored = []
    for e, v in quoted:
        en = e.lower()
        if mentions(gnorm, en) == 1:
            anchored.append((e, v))
    if len(anchored) < min_claims:
        return None, "ambiguous_entity"
    claims = anchored
    if len(claims) > max_claims:
        return None, "too_many_claims"

    gold = score_grounded_pairs(final, claims)
    if gold < 1.0:
        return None, "gold_unbound"

    text = "\n".join((m.get("content") or "") for m in msgs)
    if contaminated(text):
        return None, "eval_fact"

    prompt_msgs = msgs[:-1]          # everything up to the answer being mined
    plen = prompt_len_fn(prompt_msgs)
    if plen > max_prompt_tok:
        return None, "long_prompt"

    return {
        "messages": msgs,
        "prompt_msgs": prompt_msgs,
        "prompt_tokens": plen,
        "gold_answer": final,
        "claims": claims,
        "gold_score": gold,
        "n_calls": len(pp),
    }, "ok"


# ------------------------------------------------------------------ io
def stream_jsonl(path):
    """Yield dicts from .jsonl or .jsonl.zst (subprocess zstd, as
    build_multi_mix.stream_zst does — the Pi has the binary, maybe not the
    python binding)."""
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dcq", path], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        stream = p.stdout
    else:
        stream = open(path, "r")
    try:
        for raw in stream:
            raw = raw.strip()
            if raw:
                try:
                    yield json.loads(raw)
                except Exception:
                    continue
    finally:
        stream.close()
        if path.endswith(".zst"):
            p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=False,
                    help="normalized jsonl(.zst) files or globs "
                         "(default: multi-dolci-0000.jsonl.zst under --norm)")
    ap.add_argument("--norm", default=os.path.expanduser(
        "~/ai/sft-pipeline/norm-multi"))
    ap.add_argument("--out", default="dpo-mined")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--max-rows", type=int, default=2500,
                    help="cap on emitted rows (--scan-cap is for input rows)")
    ap.add_argument("--scan-cap", type=int, default=0)
    ap.add_argument("--min-claims", type=int, default=MIN_CLAIMS)
    ap.add_argument("--max-claims", type=int, default=MAX_CLAIMS)
    ap.add_argument("--max-prompt-tok", type=int, default=MAX_PROMPT_TOK)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    paths = list(args.inputs or [])
    if not paths:
        paths = sorted(glob.glob(os.path.join(args.norm, "multi-dolci-*.jsonl.zst")))
    if not paths:
        print(f"no input files (looked in {args.norm})", file=sys.stderr)
        return 2
    paths = [p for pat in paths for p in sorted(glob.glob(pat))] if any(
        "*" in p for p in paths) else paths

    tok_path = args.tokenizer or os.environ.get("TOK_JSON", "")
    if not tok_path or not os.path.exists(tok_path):
        cands = [os.path.join(args.norm, "tokenizer.json"),
                 os.path.join(os.path.dirname(args.norm), "tokenizer.json")]
        tok_path = next((c for c in cands if os.path.exists(c)), "")
    if tok_path:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        cache = {}

        # Approximate the rendered length per message and memoize by content
        # hash: the system prompt (full tool schemas, 300-1500 tokens) is the
        # same string across most rows of a shard, so this is nearly free. It is
        # an approximation of `rollout.render_prefix`, not a byte-exact count —
        # the length gate has slack for that, and dpo_pairs.py re-measures
        # exactly for bucketing.
        def msg_cost(m):
            role, content = m["role"], (m.get("content") or "")
            if role == "tool":
                body = "<|tool_result|>" + content
            elif role == "assistant":
                body = content or ""
                for tc in (m.get("tool_calls") or []):
                    body += "<|tool_call|>" + json.dumps(
                        {"name": tc["name"], "arguments": tc["arguments"]},
                        ensure_ascii=False)
                body += "<|im_end|>"
            else:
                body = content
            key = (role, body)
            if key not in cache:
                cache[key] = len(tok.encode(
                    f"<|im_start|>{role}\n{body}<|im_end|>\n",
                    add_special_tokens=False).ids)
            return cache[key]

        def prompt_len_fn(msgs):
            return sum(msg_cost(m) for m in msgs)
    else:
        print("[mine] no tokenizer.json found; prompt length will be "
              "approximated by characters/3 (buckets will be approximate)",
              flush=True)

        def prompt_len_fn(msgs):
            return sum(len(m.get("content") or "") for m in msgs) // 3

    os.makedirs(args.out, exist_ok=True)
    drops = collections.Counter()
    kept, seen = [], set()
    scanned = 0
    per_shard, shard_i = [], 0
    lens = []

    def flush():
        nonlocal per_shard, shard_i
        if not per_shard:
            return
        path = os.path.join(args.out, f"mined-{shard_i:04d}.jsonl")
        with open(path, "w") as fh:
            for r in per_shard:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[mine] wrote {path} ({len(per_shard)} rows)", flush=True)
        per_shard, shard_i = [], shard_i + 1

    for path in paths:
        for rec in stream_jsonl(path):
            scanned += 1
            if args.scan_cap and scanned > args.scan_cap:
                break
            row, why = build_row(rec.get("messages") or [], prompt_len_fn,
                                 args.min_claims, args.max_claims,
                                 args.max_prompt_tok)
            drops[why] += 1
            if row is None:
                continue
            k = dedup_key(row["messages"])
            if k in seen:
                drops["dup"] += 1
                continue
            seen.add(k)
            row["key"] = k
            row["source"] = rec.get("source", "multi-dolci")
            lens.append(row["prompt_tokens"])
            kept.append(row)
            per_shard.append(row)
            if len(per_shard) >= args.shard_size:
                flush()
            if len(kept) >= args.max_rows:
                break
        print(f"[mine] {os.path.basename(path)}: scanned={scanned} "
              f"kept={len(kept)}", flush=True)
        if args.scan_cap and scanned > args.scan_cap:
            break
        if len(kept) >= args.max_rows:
            break
    flush()

    lens.sort()
    def pct(p):
        return lens[min(len(lens) - 1, int(len(lens) * p))] if lens else 0
    stats = {
        "scanned": scanned,
        "kept": len(kept),
        "drops": dict(drops),
        # fraction of rows that reached the gold-binding check and passed it
        "gold_yield": (len(kept) / (len(kept) + drops["gold_unbound"])
                       if (len(kept) + drops["gold_unbound"]) else 0.0),
        "gold_checked": len(kept) + drops["gold_unbound"],
        "prompt_tokens": {"p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                          "max": lens[-1] if lens else 0},
        "n_claims_mean": (sum(len(r["claims"]) for r in kept) / len(kept)
                          if kept else 0),
    }
    with open(os.path.join(args.out, "mine-manifest.json"), "w") as fh:
        json.dump(dict(args=vars(args), stats=stats), fh, indent=1)
    print(json.dumps(stats, indent=1))

    if not kept:
        print("\nno rows kept — check the drops table before anything else",
              flush=True)
        return 2
    gy = stats["gold_yield"]
    if gy < GOLD_YIELD_FLOOR:
        print(f"\nFAIL gold yield {gy:.3f} < {GOLD_YIELD_FLOOR} over "
              f"{stats['gold_checked']} rows that reached the binding check. "
              f"A healthy miner scores ~0.8 here; near 0.1 means entity "
              f"extraction is broken. Do not lower the floor.", flush=True)
        return 3
    print(f"\nMINE OK — {len(kept)} rows from {scanned} scanned "
          f"({len(kept)/max(1,scanned):.1%}), gold yield {gy:.3f} over "
          f"{stats['gold_checked']} rows that reached the binding check. "
          f"Next: rl/dpo_pairs.py", flush=True)
    return 0


# ------------------------------------------------------------------ selftest
def selftest():
    """Pinned cases for the two functions with no oracle. The rest of the gate
    is checked against gold by the caller."""
    fails = []

    def expect(name, got, want):
        ok = got == want
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {got!r} (want {want!r})")
        if not ok:
            fails.append(name)

    sysmsg = {"role": "system", "content": "tools"}
    tc = lambda n, a: {"role": "assistant", "content": "",
                       "tool_calls": [{"name": n, "arguments": json.dumps(a)}]}
    tool = lambda c: {"role": "tool", "content": json.dumps(c)}
    user = {"role": "user", "content": "coords?"}

    geo = [sysmsg, user,
           tc("geo.geocode", {"address": "Kampala"}),
           tc("geo.geocode", {"address": "Gulu"}),
           tool({"results": {"city": "Kampala", "latitude": 0.3476,
                             "longitude": 32.5825}}),
           tool({"results": {"city": "Gulu", "latitude": 2.7724,
                             "longitude": 32.2881}}),
           {"role": "assistant",
            "content": ("Kampala: lat 0.3476, lon 32.5825. "
                        "Gulu: lat 2.7724, lon 32.2881.")}]

    print("== positional_pairs reconstructs tool_name (tool msgs carry none)")
    pp = positional_pairs(geo)
    expect("pair_count", len(pp), 2)
    expect("names", [p[0] for p in pp], ["geo.geocode", "geo.geocode"])
    expect("results_ordered", [p[2]["results"]["city"] for p in pp],
           ["Kampala", "Gulu"])

    print("== floats survive as printed literals (substring grading depends on it)")
    expect("float_literal", pp[0][2]["results"]["latitude"], "0.3476")

    print("== mine_claims finds entity/value pairs")
    claims = mine_claims(pp)
    print("  claims:", claims)
    expect("has_kampala_lat", ("Kampala", "0.3476") in claims, True)
    expect("has_gulu_lat", ("Gulu", "2.7724") in claims, True)

    print("== gold binds everything -> gate passes")
    def plen(msgs):
        return 10
    row, why = build_row(geo, plen, 2, 12, 1536)
    expect("row_ok", why, "ok")
    expect("row_claims", len(row["claims"]), len(claims))
    expect("prompt_excludes_answer",
           row["prompt_msgs"][-1]["role"], "tool")

    print("== a swapped-value answer is what the scorer must catch")
    swapped = geo[:-1] + [{"role": "assistant", "content":
                           "Kampala: lat 2.7724, lon 32.5825. Gulu: lat 0.3476."}]
    s = score_grounded_pairs(swapped[-1]["content"], claims)
    print(f"  swapped grounded score: {s:.3f}")
    expect("swap_penalised", s < 1.0, True)

    print("== same_entity_set separates binding failure from a different answer")
    expect("pure_binding", same_entity_set(
        "Kampala: 0.3476. Gulu: 2.7724.", geo[-1]["content"], claims), True)
    expect("style_only", same_entity_set(
        "The coordinates are: Kampala 0.3476 / 32.5825, Gulu 2.7724 / 32.2881.",
        geo[-1]["content"], claims), True)
    expect("different_answer", same_entity_set(
        "I could not retrieve that information.", geo[-1]["content"], claims),
        False)

    print("== real-corpus formatting: the mock env never hits these")
    fmt = [{"city": "Tokyo", "temp_c": 18.4}, {"city": "New York", "temp_c": 22.1}]
    prose = [{"role": "system", "content": "t"}, user,
             tc("get_weather", {"city": "Tokyo"}),
             tc("get_weather", {"city": "New York"}),
             tool(fmt[0]), tool(fmt[1]),
             {"role": "assistant", "content":
              "Tokyo: $18.40. New York: $22.10."}]
    fclaims = mine_claims(positional_pairs(prose))
    print("  claims:", fclaims)
    expect("comma_dp_binds", score_grounded_pairs(prose[-1]["content"], fclaims),
           1.0)

    big = [sysmsg, user,
           tc("get_market", {"ids": "80057281,80189685"}),
           tc("get_rating", {"ids": "80057281,80189685"}),
           tool({"results": [{"id": "80057281", "year": 2016}]}),
           tool({"results": [{"id": "80189685", "year": 2019}]}),
           {"role": "assistant", "content":
            "Stranger Things (ID 80057281) came out in 2016. "
            "The Witcher (ID 80189685) came out in 2019."}]
    bclaims = mine_claims(positional_pairs(big))
    print("  list-arg claims:", bclaims)
    expect("list_arg_splits", sorted({e for e, _ in bclaims}),
           ["80057281", "80189685"])
    expect("list_arg_binds", score_grounded_pairs(big[-1]["content"], bclaims),
           1.0)
    expect("list_arg_row_ok", build_row(big, plen, 2, 12, 1536)[1], "ok")

    print("== an elided claim is dropped, not held against the row")
    elide = [sysmsg, user,
             tc("get_weather", {"city": "Berlin"}),
             tc("get_weather", {"city": "Lisbon"}),
             tool({"city": "Berlin", "temp_c": 12.5}),
             tool({"city": "Lisbon", "temp_c": 19.8}),
             {"role": "assistant", "content": "Berlin is 12.5C."}]
    row_e, why_e = build_row(elide, plen, 1, 12, 1536)
    expect("elided_kept", why_e, "ok")
    print("  surviving claims:", row_e["claims"])
    expect("elision_removed_lisbon",
           [c for c in row_e["claims"] if c[0] == "Lisbon"], [])

    print("== an id is not a measurement (cross-attribution)")
    idrow = [sysmsg, user,
             tc("get_market", {"ids": "80057281,80189685"}),
             tc("get_rating", {"ids": "80057281,80189685"}),
             tool({"results": [{"id": "80057281", "year": 2016, "peer": "80189685"}]}),
             tool({"results": [{"id": "80189685", "year": 2019}]}),
             {"role": "assistant", "content":
              "Stranger Things (ID 80057281) came out in 2016. "
              "The Witcher (ID 80189685) came out in 2019."}]
    ic = mine_claims(positional_pairs(idrow))
    print("  claims:", ic)
    expect("no_id_as_value",
           [c for c in ic if c[1] in ("80057281", "80189685")], [])

    print("== a repeated entity is not a binding exercise")
    # 'female' appears once per health question; the clause span between the two
    # mentions does not contain the first question's value
    amb = [sysmsg, user,
           tc("get_calories", {"profile": "female,35,moderately active"}),
           tc("get_bmi", {"profile": "female,35,moderately active"}),
           tool({"calories": 1425, "profile": "female"}),
           tool({"bmi": 24.1, "profile": "female"}),
           {"role": "assistant", "content":
            "For a 35-year-old female the target is about 1,425 calories. "
            "For a female at that weight the BMI is 24.1."}]
    expect("ambiguous_rejected", build_row(amb, plen, 2, 12, 1536)[1],
           "ambiguous_entity")

    print("== gate rejections")
    expect("one_call", build_row([sysmsg, user, tc("a", {}), tool({"x": 1}),
                                  {"role": "assistant", "content": "1"}],
                                 plen, 2, 12, 1536)[1], "one_call")
    expect("too_many_claims",
           build_row(geo, plen, 2, 1, 1536)[1], "too_many_claims")
    dangling = geo[:4] + [{"role": "assistant", "content": "ok"}]
    expect("dangling", build_row(dangling, plen, 2, 12, 1536)[1], "pairing")
    orphan = geo[:4] + [tool({"results": {"city": "X", "latitude": 1.0}}),
                        {"role": "assistant", "content": "X 1.0"}]
    expect("orphan_result", build_row(orphan, plen, 2, 12, 1536)[1], "pairing")

    print("== eval-fact decontam runs on new text (row binds its claims first,")
    print("   so the decontam gate is what rejects it, not gold_unbound)")
    leaky_msgs = [sysmsg, user,
                  tc("geo.geocode", {"address": "Tokyo"}),
                  tc("get_weather", {"city": "New York"}),
                  tool({"results": {"city": "Tokyo", "temp_c": 18.4}}),
                  tool({"results": {"city": "New York", "temp_c": 22.1}}),
                  {"role": "assistant",
                   "content": "Tokyo is 18.4C and New York is 22.1C."}]
    expect("leaky_binds_first", score_grounded_pairs(leaky_msgs[-1]["content"],
           mine_claims(positional_pairs(leaky_msgs))), 1.0)
    expect("eval_fact", build_row(leaky_msgs, plen, 2, 12, 1536)[1],
           "eval_fact")

    print("== length gate")
    expect("long_prompt", build_row(geo, lambda m: 9999, 2, 12, 1536)[1],
           "long_prompt")

    print()
    if fails:
        print(f"MINE SELFTEST FAIL: {fails}")
        return 1
    print("MINE SELFTEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())