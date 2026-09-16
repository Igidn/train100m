import json

tot = {"p1": 0, "p2": 0}
rows = []
manifests = {}
for suf in ("", "-w1", "-w2", "-w3", "-w4", "-p2a", "-p2b", "-p2c", "-p2d"):
    try:
        m = json.load(open(f"manifest{suf}.json"))
    except OSError:
        continue
    for k, v in m.get("sources", {}).items():
        manifests[k] = v
    try:
        s = json.load(open(f"state{suf}.json"))
    except OSError:
        continue
    for k, v in s["entries"].items():
        est = round(sum(v.get("chars", {}).values()) / 3.93 / 1e6)
        for p, c in v.get("chars", {}).items():
            tot[p] += c
        rows.append((k, v.get("status"), est))
rows.sort()
for k, st, est in rows:
    print(f"{k:38s} {str(st):10s} {est:6d}M est tok")

print(f"\nphase1 pool: {round(tot['p1']/3.93/1e6)}M est tokens "
      "(will be sampled down to exactly 1500M)")
print(f"phase2 pool: {round(tot['p2']/3.93/1e6)}M est tokens "
      "(will be sampled down to exactly 500M)")

merged = {"sources": manifests, "pool": 2.5,
          "chars_per_token_assumed": 3.93,
          "missing_notes": json.load(open("data_sources.py") and open(
              "missing_notes.json")) if False else
              __import__("data_sources").MISSING_NOTES,
          "fetch_order": list(manifests.keys())}
json.dump(merged, open("manifest.json", "w"), indent=1)
print("\nmerged manifest.json written:",
      f"{len(manifests)} sources")
