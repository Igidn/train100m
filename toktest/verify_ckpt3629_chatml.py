"""Follow-up: ChatML special-token tests + generation for checkpoint-3629."""
import os, sys, time
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama.model import LLaMA

torch.set_num_threads(os.cpu_count() or 8)
torch.manual_seed(0)
CKPT = os.path.expanduser("~/Downloads/checkpoint-3629.pt")
EOS, IM_START, IM_END = 0, 1, 2

tok = Tokenizer.from_file("tokenizer.json")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
              n_kv_heads=4, head_dim=64, ffn_dim=2048)
model.load_state_dict(ck["model"], strict=True); model.eval(); del ck

tokens = np.memmap("val-000.bin", dtype=np.uint16, mode="r")
off = np.load("val-000.offsets.npy")
docs = [tokens[off[i]:off[i+1]] for i in range(len(off)-1)]
cand = [d for d in docs if len(d) >= 600][:40]

def logits_of(ctx):
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(ctx, dtype=np.int64))[None])[0, -1]

def rank_of(lg, tid):
    return (lg > lg[tid]).sum().item()

print("== ChatML specials ==", flush=True)
d0 = cand[0]
for name, ctx in [("doc end", d0[-513:-1]), ("doc mid", d0[300:804])]:
    lg = logits_of(ctx)
    p = torch.softmax(lg.float(), -1)
    print(f"{name}: rank eos {rank_of(lg, EOS)}, rank im_end {rank_of(lg, IM_END)}, "
          f"p(eos) {p[EOS]:.4f}, p(im_end) {p[IM_END]:.8f}, p(im_start) {p[IM_START]:.8f}",
          flush=True)

print("\n== ChatML conditioning ==", flush=True)
for prompt in ["<|im_start|>user\n", "<|im_start|>assistant\n",
               "<|im_start|>assistant\n"]:
    ctx = list(map(int, d0[:400])) + tok.encode(prompt).ids
    lg = logits_of(ctx)
    top = torch.softmax(lg.float(), -1).topk(5)
    pairs = [(repr(tok.decode([int(i)])), float(pr)) for pr, i in zip(top.values, top.indices)]
    print(f"ctx {prompt!r} -> top5 {pairs}", flush=True)
    if prompt.endswith("\n"):
        imr = rank_of(lg, IM_END)
        print(f"    im_end rank: {imr}, p {float(torch.softmax(lg.float(),-1)[IM_END]):.6f}", flush=True)

# pure ChatML context (no doc prefix): continuation after assistant role
ctx = tok.encode("<|im_start|>assistant\n").ids
lg = logits_of(ctx)
top = torch.softmax(lg.float(), -1).topk(8)
print("pure '<|im_start|>assistant\\n' -> top8",
      [(repr(tok.decode([int(i)])), float(pr)) for pr, i in zip(top.values, top.indices)], flush=True)

print("\n== im_end placement: does model emit <|im_end|> after assistant answer? ==", flush=True)
# build a complete ChatML exchange in-context, then check next-token preference
q = "What is the capital of France?"
a = "The capital of France is Paris."
seq = "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n" + a
ids = list(tok.encode(seq).ids)
lg = logits_of(ids)
p = torch.softmax(lg.float(), -1)
print(f"after complete assistant turn: rank im_end {rank_of(lg, IM_END)}, "
      f"rank eos {rank_of(lg, EOS)}, p(im_end) {p[IM_END]:.4f}, p(eos) {p[EOS]:.6f}", flush=True)
# after im_end + newline, should prefer <|im_start|> or eos
ids2 = ids + tok.encode("<|im_end|>\n").ids
lg2 = logits_of(ids2)
p2 = torch.softmax(lg2.float(), -1)
print(f"after '<|im_end|>\\n': rank im_start {rank_of(lg2, IM_START)}, "
      f"rank eos {rank_of(lg2, EOS)}, p(im_start) {p2[IM_START]:.4f}, p(eos) {p2[EOS]:.4f}", flush=True)

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
