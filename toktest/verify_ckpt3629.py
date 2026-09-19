"""Checkpoint-3629 verification: structure, EOS placement, next-phase readiness.

Run from toktest/:  python verify_ckpt3629.py [quick|full]
"""
import math, os, sys, time
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama.model import LLaMA

torch.set_num_threads(os.cpu_count() or 8)
torch.manual_seed(0)

CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Downloads/checkpoint-3629.pt")
MODE = sys.argv[2] if len(sys.argv) > 2 else "quick"
EOS, SEQ = 0, 2048

tok = Tokenizer.from_file("tokenizer.json")
IM_START, IM_END = 1, 2

print("== loading checkpoint ==", flush=True)
t0 = time.time()
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"step {ck['step']} phase {ck['phase']} micros {ck['micros_done']} "
      f"tokens {ck['tokens_seen']/1e9:.4f}B  (load {time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- structure
model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
              n_kv_heads=4, head_dim=64, ffn_dim=2048)
missing, unexpected = model.load_state_dict(ck["model"], strict=True), None
model.eval()
print(f"state dict: strict load OK, params {model.num_params()/1e6:.1f}M", flush=True)
w = model.lm_head.weight
print(f"tied head: lm_head.weight is tok_emb.weight -> {w is model.tok_emb.weight}", flush=True)

opt = ck["opt"]
nnan = sum(1 for s in opt["state"].values()
           if not torch.isfinite(s["exp_avg"]).all() or not torch.isfinite(s["exp_avg_sq"]).all())
nparam = sum(1 for s in opt["state"].values() if s.get("exp_avg") is not None)
steps_opt = [s.get("step") for s in opt["state"].values() if s.get("step") is not None]
print(f"opt: {nparam} states, non-finite: {nnan}, "
      f"step {steps_opt[0] if steps_opt else '?'} (ckpt step {ck['step']})", flush=True)
del opt
gc = {k: v.abs().max().item() for k, v in ck["model"].items() if torch.is_tensor(v)}
wmax = max(gc.values())
print(f"weight abs-max across tensors: {wmax:.2f} "
      f"({'OK for fp16' if wmax < 100 else 'WARN: large'})", flush=True)

# ---------------------------------------------------------------- data
tokens = np.memmap("val-000.bin", dtype=np.uint16, mode="r")
off = np.load("val-000.offsets.npy")
docs = [tokens[off[i]:off[i+1]] for i in range(len(off)-1)]
cand = [d for d in docs if len(d) >= 600][:40]

def logits_of(ctx):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(ctx, dtype=np.int64))[None])[0, -1]

def rank_of(logits, tid):
    return (logits > logits[tid]).sum().item()

# ---------------------------------------------------------------- val loss
N_EVAL = 8 if MODE == "quick" else 32
print(f"\n== val loss ({N_EVAL} chunks x {SEQ}) ==", flush=True)
tot, n = 0.0, 0
with torch.no_grad():
    for c in range(N_EVAL):
        b = torch.from_numpy(np.asarray(tokens[c*SEQ:(c+1)*SEQ], dtype=np.int64))[None]
        tot += model(b[:, :-1], b[:, 1:]).item(); n += 1
        print(f"  chunk {c}: {tot/n:.4f}", flush=True)
vl = tot / n
print(f"val loss: {vl:.4f}  ppl {math.exp(vl):.1f}", flush=True)

# ---------------------------------------------------------------- EOS @ doc end
print("\n== eos prediction at doc ends (ctx = 512 tok before final eos) ==", flush=True)
ranks, end_p = [], []
for d in cand:
    lg = logits_of(d[-513:-1])
    ranks.append(rank_of(lg, EOS))
    end_p.append(torch.softmax(lg.float(), -1)[EOS].item())
ranks = np.array(ranks)
print(f"eos rank: median {np.median(ranks):.0f} | top1 {(ranks==0).sum()}/{len(cand)} "
      f"top5 {(ranks<5).sum()}/{len(cand)} top20 {(ranks<20).sum()}/{len(cand)}", flush=True)
print(f"eos prob at end: mean {np.mean(end_p):.3f} min {np.min(end_p):.4f} "
      f"max {np.max(end_p):.4f}", flush=True)

# ---------------------------------------------------------------- EOS mid-doc (false positives)
print("\n== eos rank at random mid-doc positions ==", flush=True)
rng = np.random.default_rng(7)
mid_r, mid_p = [], []
for _ in range(40):
    d = cand[rng.integers(len(cand))]
    p = int(rng.integers(512, len(d) - 100))
    lg = logits_of(d[p - 512:p])
    mid_r.append(rank_of(lg, EOS))
    mid_p.append(torch.softmax(lg.float(), -1)[EOS].item())
mid_r = np.array(mid_r)
print(f"eos rank: median {np.median(mid_r):.0f} top1 {(mid_r==0).sum()}/40 "
      f"mean prob {np.mean(mid_p):.4f}", flush=True)

# ---------------------------------------------------------------- eos prob vs distance to end
print("\n== eos prob at end-k (single forward per doc) ==", flush=True)
probs = np.zeros((len(cand), 33))
with torch.no_grad():
    for j, d in enumerate(cand):
        ctx = torch.from_numpy(np.asarray(d[-513:-1], dtype=np.int64))[None]
        lg = model(ctx)[0]
        p = torch.softmax(lg[-33:].float(), -1)[:, EOS].numpy()
        probs[j] = p[::-1]
for k in (1, 2, 3, 5, 8, 12, 16, 24, 32):
    print(f"k={k:2d}: eos prob {probs[:, k-1].mean():.4f}", flush=True)

# ---------------------------------------------------------------- greedy rollout near true end
print("\n== greedy rollout: doc body (no eos) truncated at last k, steps to eos ==", flush=True)
for k in (1, 3, 10):
    hits, misses = [], 0
    for d in cand[:10]:
        body = d[:-1][:-k]
        ctx = list(map(int, body[-256:]))
        hit = None
        with torch.no_grad():
            for i in range(k + 8):
                lg = model(torch.tensor([ctx[-512:]], dtype=torch.int64))[0, -1]
                nx = int(lg.argmax())
                if nx == EOS:
                    hit = i
                    break
                ctx.append(nx)
        if hit is None:
            misses += 1
        else:
            hits.append(hit)
    print(f"k={k:2d}: eos {len(hits)}/10, delays {hits}, no-eos {misses}", flush=True)

# ---------------------------------------------------------------- ChatML exposure
print("\n== ChatML specials (im_start/im_end) ==", flush=True)
def r2(a, b):
    return (a > b).sum().item()
d0 = cand[0]
lg_end = logits_of(d0[-513:-1])            # doc end: <|im_end|> should NOT beat eos
lg_mid = logits_of(d0[300:804])
print(f"doc end : rank eos {r2(-lg_end, -lg_end[EOS])}, rank im_end {r2(-lg_end, -lg_end[IM_END])}")
print(f"          p(eos) {torch.softmax(lg_end.float(),-1)[EOS]:.4f} "
      f"p(im_end) {torch.softmax(lg_end.float(),-1)[IM_END]:.6f}")
print(f"doc mid : p(eos) {torch.softmax(lg_mid.float(),-1)[EOS]:.4f} "
      f"p(im_end) {torch.softmax(lg_mid.float(),-1)[IM_END]:.6f}")

# conditioned on ChatML context from phase-2 data
ctx = tok.encode("<|im_start|>user\n").ids
dctx = list(map(int, d0[:400])) + ctx
lg_c = logits_of(dctx)
top = torch.softmax(lg_c.float(), -1).topk(5)
print("after '<|im_start|>user\\n' (400-tok doc prefix): top5",
      [(repr(tok.decode([i])), round(p, 4)) for p, i in zip(top.values, top.indices)], flush=True)

# ---------------------------------------------------------------- generation
print("\n== free generation from 64-tok doc prefix ==", flush=True)
d = cand[4]
ctx = list(map(int, d[:64])); gen = []
with torch.no_grad():
    for i in range(120):
        lg = model(torch.tensor([ctx[-512:]], dtype=torch.int64))[0, -1]
        nx = int(lg.argmax())
        if nx == EOS:
            print("  [eos] at step", i, flush=True); break
        ctx.append(nx); gen.append(nx)
print(" ", repr(tok.decode(gen))[:300], flush=True)

lg = logits_of(cand[0][-513:-1].tolist())
print(f"\nlogit health: min {lg.min().item():.1f} max {lg.max().item():.1f} "
      f"finite {torch.isfinite(lg).all().item()}", flush=True)
print("ALL DONE", flush=True)
