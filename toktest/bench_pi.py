"""Quick benchmark on the Pi: pick val-eval batch size, measure decode step."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama.model import LLaMA

torch.set_num_threads(4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
              n_kv_heads=4, head_dim=64, ffn_dim=2048)
model.eval()
print(f"threads {torch.get_num_threads()}, model {(sum(p.numel() for p in model.parameters()))/1e6:.0f}M-nominal",
      flush=True)

def bench(bs, seq, reps=1):
    x = torch.randint(0, 49154, (bs, seq + 1))
    with torch.inference_mode():
        model(x[:, :seq], x[:, 1:])            # warmup
        t0 = time.time()
        for _ in range(reps):
            l = model(x[:, :seq], x[:, 1:])
        dt = time.time() - t0
    print(f"bs {bs} seq {seq}: {dt:.1f}s  ({dt/(bs*seq)*1000:.1f} ms/tok, "
          f"loss {l.item():.3f})", flush=True)

for bs in (1, 2, 4, 8):
    bench(bs, 2048)

# tiny decode-step cost (full forward, ctx=1)
x = torch.randint(0, 49154, (1, 2))
with torch.inference_mode():
    t0 = time.time()
    for _ in range(20):
        model(x[:, :1], x[:, 1:])
    print(f"1-tok fwd: {(time.time()-t0)/20*1000:.1f} ms", flush=True)
