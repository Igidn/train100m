"""Checkpoint-1757 inspection & behavioral tests (CPU)."""
import math, os, sys, time
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama.model import LLaMA

torch.set_num_threads(os.cpu_count() or 8)
torch.manual_seed(0)

tok = Tokenizer.from_file("tokenizer.json")
EOS = 0
SEQ = 2048

print("== loading checkpoint ==", flush=True)
ck = torch.load("/home/lupa/Desktop/train100m/checkpoint-1757.pt",
                map_location="cpu", weights_only=False)
print(f"step {ck['step']} phase {ck['phase']} tokens {ck['tokens_seen']/1e9:.3f}B", flush=True)

model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
              n_kv_heads=4, head_dim=64, ffn_dim=2048)
missing, unexpected = model.load_state_dict(ck["model"], strict=True), None
model.eval()

# optimizer state sanity
opt = ck["opt"]
nzero = sum(1 for s in opt["state"].values() if s.get("exp_avg") is not None
            and s["exp_avg"].abs().sum().item() == 0)
nnan = sum(1 for s in opt["state"].values()
           if not torch.isfinite(s["exp_avg"]).all() or not torch.isfinite(s["exp_avg_sq"]).all())
print(f"opt states: {len(opt['state'])} tensors, all-zero exp_avg: {nzero}, non-finite: {nnan}")
print(f"opt step count: {list(opt['state'].values())[0].get('step')}")
del ck, opt

# ---------------------------------------------------------------- val loss
tokens = np.memmap("val-000.bin", dtype=np.uint16, mode="r")
N_EVAL = 8
print(f"\n== val loss on {N_EVAL} chunks x {SEQ} tok ==", flush=True)
t0 = time.time()
tot, n = 0.0, 0
with torch.no_grad():
    for c in range(N_EVAL):
        b = torch.from_numpy(np.asarray(tokens[c*SEQ:(c+1)*SEQ], dtype=np.int64))[None]
        loss = model(b[:, :-1], b[:, 1:])
        tot += loss.item(); n += 1
        print(f"  chunk {c}: {loss.item():.4f}", flush=True)
print(f"val loss: {tot/n:.4f}  (ppl {math.exp(tot/n):.1f})  [{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- EOS tests
off = np.load("val-000.offsets.npy")
docs = [tokens[off[i]:off[i+1]] for i in range(len(off)-1)]
print(f"\ndocs: {len(docs)}, mean len {np.mean([len(d) for d in docs]):.0f}")

def logits_of(ctx):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(ctx, dtype=np.int64))[None])[0, -1]

def rank_of(logits, tid):
    return (logits > logits[tid]).sum().item()

# (a) at true doc end: does the model predict EOS?
print("\n== EOS prediction at doc ends (context = last 512 tok before eos) ==", flush=True)
cand = [d for d in docs if len(d) >= 600][:40]
ranks, top1 = [], 0
for d in cand:
    lg = logits_of(d[-513:-1])          # next-token target is the doc's eos
    r = rank_of(lg, EOS)
    ranks.append(r)
    top1 += (r == 0)
ranks = np.array(ranks)
print(f"eos rank: median {np.median(ranks):.0f}, top1 {top1}/{len(cand)}, "
      f"top5 {(ranks<5).sum()}/{len(cand)}, top20 {(ranks<20).sum()}/{len(cand)}")

# (b) mid-doc control: how often does eos get top rank spuriously?
print("\n== eos rank at random mid-doc positions (false-positive check) ==", flush=True)
rng = np.random.default_rng(7)
mid_ranks, mid_top1 = [], 0
mid_probs = []
for _ in range(40):
    d = cand[rng.integers(len(cand))]
    p = int(rng.integers(512, len(d) - 100))
    lg = logits_of(d[p - 512:p])
    mid_ranks.append(rank_of(lg, EOS))
    mid_probs.append(torch.softmax(lg.float(), -1)[EOS].item())
    mid_top1 += (mid_ranks[-1] == 0)
mid_ranks = np.array(mid_ranks)
print(f"eos rank: median {np.median(mid_ranks):.0f}, top1 {mid_top1}/40, "
      f"mean prob {np.mean(mid_probs):.4f} (data eos freq ~1/{np.mean([len(d) for d in docs]):.0f})")

# (c) eos prob at doc end for the same docs
end_probs = []
for d in cand[:20]:
    lg = logits_of(d[-513:-1])
    end_probs.append(torch.softmax(lg.float(), -1)[EOS].item())
print(f"eos prob at doc end: mean {np.mean(end_probs):.3f}, "
      f"min {np.min(end_probs):.4f}, max {np.max(end_probs):.4f}")

# (d) early-exit test: feed a full doc minus its last K real tokens; eos should
#     appear only near the true end
print("\n== early-exit: greedy gen from doc prefix, where does eos fire? ==", flush=True)
for d in cand[:3]:
    body = d[:-1]                        # strip eos
    ctx = list(body[:256])               # from 256-token prefix
    emitted = None
    with torch.no_grad():
        for i in range(140):
            lg = model(torch.tensor([ctx[-512:]], dtype=torch.int64))[0, -1]
            nx = int(lg.argmax())
            if nx == EOS:
                emitted = i
                break
            ctx.append(nx)
    print(f"  doc len {len(body)}: eos emitted at step {emitted} "
          f"({len(ctx)-256} tokens generated)" if emitted is not None
          else f"  doc len {len(body)}: no eos in 140 steps")
    if emitted is not None:
        print(f"    text tail: ...{tok.decode(ctx[-25:])!r}")

# (e) free generation right after eos (doc start conditioning)
print("\n== free generation from 64-tok doc prefix ==", flush=True)
d = cand[4]
ctx = list(d[:64])
gen = []
with torch.no_grad():
    for i in range(120):
        lg = model(torch.tensor([ctx[-512:]], dtype=torch.int64))[0, -1]
        nx = int(lg.argmax())
        if nx == EOS:
            print("  [eos] at gen step", i)
            break
        ctx.append(nx); gen.append(nx)
print("  sample:", repr(tok.decode(gen))[:300])

# (f) logit health
lg = logits_of(cand[0][-513:-1].tolist())
print(f"\n== logit health: min {lg.min().item():.1f} max {lg.max().item():.1f} "
      f"finite {torch.isfinite(lg).all().item()}")
p = torch.softmax(lg.float(), -1)
print(f"top5 prob: {p.topk(5).values.tolist()}")
print("top5 tokens:", [repr(tok.decode([i])) for i in p.topk(5).indices.tolist()])
