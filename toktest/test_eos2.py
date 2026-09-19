import os, sys, time
import numpy as np, torch
from tokenizers import Tokenizer
sys.path.insert(0, "/home/lupa/Desktop/train100m")
from llama.model import LLaMA

torch.set_num_threads(min(8, os.cpu_count() or 8))
tok = Tokenizer.from_file("tokenizer.json"); EOS = 0

t0 = time.time()
print("loading ckpt...", flush=True)
ck = torch.load("/home/lupa/Desktop/train100m/checkpoint-1757.pt", map_location="cpu", weights_only=False)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12, n_kv_heads=4, head_dim=64, ffn_dim=2048)
model.load_state_dict(ck["model"]); model.eval(); del ck

tokens = np.memmap("val-000.bin", dtype=np.uint16, mode="r")
off = np.load("val-000.offsets.npy")
docs = [tokens[off[i]:off[i+1]] for i in range(len(off)-1)]
cand = [d for d in docs if len(d) >= 600][:40]
print(f"docs {len(cand)}", flush=True)

# ---- eos prob vs distance to doc end (single forward per doc) ----
print("\n== eos prob at position end-k (k = 1..32) ==", flush=True)
probs = np.zeros((len(cand), 33))
with torch.no_grad():
    for j, d in enumerate(cand):
        ctx = torch.from_numpy(np.asarray(d[-513:-1], dtype=np.int64))[None]
        lg = model(ctx)[0]                       # (513, V) logits
        p = torch.softmax(lg[-33:].float(), -1)[:, EOS].numpy()
        probs[j] = p[::-1]                       # index 0 -> k=1
        if j % 10 == 0:
            print(f"  doc {j} done", flush=True)
ks = np.arange(1, 33)
print("k : mean eos prob")
for k in (1, 2, 3, 5, 8, 12, 16, 24, 32):
    print(f"{k:2d} : {probs[:, k-1].mean():.4f}")

# ---- batched greedy rollout: context = doc[:-k], see steps until eos ----
print("\n== batched greedy rollout (ctx 256) ==", flush=True)
ND = 10
for k in (1, 3, 10):
    rows, lens = [], []
    for d in cand[:ND]:
        body = d[:-1][:-k]
        tail = body[-256:] if len(body) >= 256 else body
        rows.append(list(map(int, tail))); lens.append(len(rows[-1]))
    L = max(lens)
    hits = [None]*ND
    done = 0
    with torch.no_grad():
        for step in range(k + 6):
            x = torch.full((ND, L), EOS, dtype=torch.int64)
            for i, r in enumerate(rows):
                x[i, -len(r):] = torch.tensor(r[-L:])
            lg = model(x)
            for i in range(ND):
                if hits[i] is not None: continue
                nx = int(lg[i, min(len(rows[i])-1, L-1)].argmax())  # last real position
                if nx == EOS:
                    hits[i] = step
                else:
                    rows[i].append(nx)
            done = sum(h is not None for h in hits)
            if done == ND: break
    hs = [h for h in hits if h is not None]
    print(f"k={k:2d}: eos hit {len(hs)}/{ND}, delays {hs}", flush=True)

# ---- sampled generation (t=0.8, top-p .95, 80 tokens) ----
print("\n== sampled gen t=0.8 ==", flush=True)
d = cand[6]; ctx = list(map(int, d[:128])); gen = []
torch.manual_seed(1)
with torch.no_grad():
    for i in range(80):
        lg = model(torch.tensor([ctx[-256:]], dtype=torch.int64))[0, -1].float() / 0.8
        pr = torch.softmax(lg, -1)
        sp, si = torch.sort(pr, descending=True)
        cum = torch.cumsum(sp, -1); keep = cum - sp < 0.95
        sp, si = sp[keep], si[keep]
        nx = int(si[torch.multinomial(sp, 1)])
        if nx == EOS:
            print("[eos] step", i, flush=True); break
        ctx.append(nx); gen.append(nx)
print(repr(tok.decode(gen))[:400], flush=True)
print("ALL DONE", flush=True)
