#!/usr/bin/env python3
"""Does stating the per-command required arguments in the system prompt make the
policy emit complete editor calls? (round-6 probe, throwaway)

The tool schema can only declare `required: [command, path]` — `required` is a
property of the whole parameter object, not per `command` value. The model emits
exactly what the schema asks for, so a `str_replace` with no `old_str`/`new_str`
is schema-*correct* and tool-*invalid*. This measures the valid-edit rate under
the two prompts on the same tasks, same model, same temperature.

    python rl/prompt_probe.py --tasks 8 --group 4 --model tinyballs-dpo
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import agent_env as A  # noqa: E402
from agent_rollout import OllamaPolicy  # noqa: E402

OLD = A.system_prompt

EXTRA = ("Tool arguments are required, not optional: `str_replace` needs "
         "`old_str` and `new_str`; `create` needs `file_text`; `insert` needs "
         "`new_str` and `insert_line`; `view` needs `path`. A call that omits a "
         "required argument is rejected and changes nothing.\n")


def new_prompt(workdir: str) -> str:
    s = OLD(workdir)
    anchor = "for file edits.\nAvailable tools:"
    if anchor not in s:
        raise SystemExit("prompt anchor moved; update the probe")
    return s.replace(anchor, "for file edits.\n" + EXTRA + "Available tools:")


def probe(prompt_fn, tasks, model, gen, temp, label):
    A.system_prompt = prompt_fn
    pol = OllamaPolicy(model, temp=temp)
    st = dict(label=label, eps=0, sr_calls=0, sr_applied=0, sr_rejected=0,
              eps_with_applied=0, eps_with_mutating_call=0, passes=0, steps=0)
    for t in tasks:
        for _ in range(gen):
            ep = A.run_episode(t, pol)
            st["eps"] += 1
            st["steps"] += ep["steps"]
            applied = False
            mutated = False
            for name, args, res in A.calls_with_results(ep["messages"]):
                if name == "str_replace_editor" and args.get("command") in (
                        "str_replace", "create", "insert"):
                    mutated = True
                    if args.get("command") == "str_replace":
                        st["sr_calls"] += 1
                        if A.mutation_applied(args, res):
                            st["sr_applied"] += 1
                        else:
                            st["sr_rejected"] += 1
                    applied = applied or A.mutation_applied(args, res)
            st["eps_with_applied"] += int(applied)
            st["eps_with_mutating_call"] += int(mutated)
            st["passes"] += int(ep["check"]["passed"])
    A.system_prompt = OLD
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--model", default="tinyballs-dpo")
    ap.add_argument("--temp", type=float, default=1.0)
    a = ap.parse_args()

    tasks = [A.generate_task(a.seed0 + i) for i in range(a.tasks)]
    out = []
    for fn, label in ((OLD, "current"), (new_prompt, "with-required-args")):
        st = probe(fn, tasks, a.model, a.group, a.temp, label)
        out.append(st)
        print(json.dumps(st), flush=True)
    print()
    for st in out:
        print(f"{st['label']:<20} applied-edit episodes {st['eps_with_applied']}/{st['eps']}  "
              f"str_replace applied {st['sr_applied']}/{st['sr_calls']}  "
              f"passes {st['passes']}  mean steps {st['steps']/max(1,st['eps']):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
