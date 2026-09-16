import json
import collections

P1 = {"web": 670, "books": 150, "wikipedia": 130, "code": 120, "math": 110,
      "synthetic": 80, "papers": 80, "qa": 80, "micro": 80}
P2 = {"anneal_instructional": 100, "anneal_math_quality": 90,
      "anneal_code_quality": 80, "anneal_math_solutions": 70,
      "anneal_wiki_qa": 70, "anneal_oer_ocr": 40, "anneal_reasoning": 25,
      "anneal_agentic": 25}

dom = collections.defaultdict(lambda: {"p1": 0, "p2": 0})
m = json.load(open("manifest.json"))
for k, v in m["sources"].items():
    d = v["domain"]
    dom[d]["p1"] += v["chars"].get("p1", 0)
    dom[d]["p2"] += v["chars"].get("p2", 0)

print(f"{'domain':24s} {'p1 est':>8s} {'p1 tgt':>8s} {'p2 est':>8s} {'p2 tgt':>8s}")
for d in sorted(dom):
    e1 = round(dom[d]["p1"] / 3.93 / 1e6)
    e2 = round(dom[d]["p2"] / 3.93 / 1e6)
    print(f"{d:24s} {e1:6d}M {P1.get(d, 0):6d}M {e2:6d}M {P2.get(d, 0):6d}M")
