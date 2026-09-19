"""One-time: strip checkpoint-final.pt into an mmap-friendly model-only file.

The full ckpt (model + adam state + rng, 1.4GB anon) OOM-kills a 4GB Pi.
This runs the optimizer/structure checks once, saves them to JSON, and
writes <dst> containing only {model, step, phase, micros_done, tokens_seen}
so verify_final.py can torch.load(mmap=True, weights_only=True) it.

Run:  python strip_ckpt.py <full-ckpt.pt> <model-only.pt>
"""
import json
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]

print(f"loading {src} ...", flush=True)
ck = torch.load(src, map_location="cpu", weights_only=False)
print(f"step {ck['step']} phase {ck['phase']} micros {ck['micros_done']} "
      f"tokens {ck['tokens_seen']/1e9:.4f}B", flush=True)

summary = dict(step=ck["step"], phase=ck["phase"],
               micros_done=ck["micros_done"], tokens_seen=ck["tokens_seen"])

opt = ck["opt"]
n1 = sum(1 for s in opt["state"].values()
         if "exp_avg" in s and not torch.isfinite(s["exp_avg"]).all())
n2 = sum(1 for s in opt["state"].values()
         if "exp_avg_sq" in s and not torch.isfinite(s["exp_avg_sq"]).all())
steps = [s.get("step") for s in opt["state"].values() if s.get("step") is not None]
summary["opt"] = dict(nstates=len(opt["state"]),
                      nonfinite_exp_avg=n1, nonfinite_exp_avg_sq=n2,
                      opt_step=float(steps[0]) if steps else None,
                      lr=float(opt["param_groups"][0]["lr"]),
                      betas=[float(b) for b in opt["param_groups"][0].get("betas", ())],
                      weight_decay=float(opt["param_groups"][0].get("weight_decay", 0)))
del opt

wmax = max(v.abs().max().item() for v in ck["model"].values())
nbad = sum(1 for v in ck["model"].values() if not torch.isfinite(v).all())
summary["weight_absmax"] = wmax
summary["nonfinite_model_tensors"] = nbad
summary["rng_keys"] = [k for k in ("rng", "np_rng", "py_rng") if k in ck]
summary["model_keys"] = len(ck["model"])

out = dict(model=ck["model"], step=ck["step"], phase=ck["phase"],
           micros_done=ck["micros_done"], tokens_seen=ck["tokens_seen"])
ck.clear()
print(f"saving {dst} ...", flush=True)
torch.save(out, dst)
with open(dst.replace(".pt", "-meta.json"), "w") as f:
    json.dump(summary, f, indent=1)
print("done:", json.dumps(summary), flush=True)
