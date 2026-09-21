"""On-GPU A/B benchmark for the run-2 throughput levers, no full run needed.

Times forward+backward per micro at production shapes (micro 8 x seq 2048)
for each KDA_IMPL / ATTN_IMPL / checkpoint combination and prints tok/s per
combo plus peak VRAM. Run it in a throwaway Kaggle session first; whatever
it says, believe that over any estimate:

    accelerate launch kda/bench.py            # default matrix
    MICRO_BS=16 kda/bench.py                  # probe bigger micros
    KDA_IMPL=torch kda/bench.py               # skip fla (and its install)

Exit code 0 even when individual combos OOM — each is reported as skipped.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama.kda import resolve_kda_impl
from llama.model import HybridModel

SEQ = int(os.environ.get("SEQ_LEN", 2048))
RUNS = int(os.environ.get("RUNS", 5))
torch.backends.cudnn.benchmark = True

# resolve the KDA backend once (validates fla against the torch scan)
impl_req = os.environ.get("KDA_IMPL", "auto")
kda_impl = resolve_kda_impl(impl_req, torch.device("cuda") if torch.cuda.is_available()
                            else torch.device("cpu"))


def bench(micro_bs, kda_impl, attn_impl, ckpt_attn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = HybridModel(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
                        n_kv_heads=4, head_dim=64, ffn_dim=2048,
                        kda_impl=kda_impl, attn_impl=attn_impl)
    for blk in model.blocks:
        blk.checkpoint_attn = ckpt_attn
    model = model.half().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, 49154, (micro_bs, SEQ + 1), device="cuda")
    scaler = torch.amp.GradScaler("cuda")

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x[:, :-1], x[:, 1:])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    try:
        step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(RUNS):
            step()
        torch.cuda.synchronize()
        dt = (time.time() - t0) / RUNS
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None
    tok_s = micro_bs * SEQ / dt
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt
    torch.cuda.empty_cache()
    return tok_s, peak


print(f"seq {SEQ}, {RUNS} timed runs per combo\n", flush=True)
print(f"{'micro':>5} {'kda':>6} {'attn':>10} {'ckpt':>5} "
      f"{'tok/s':>8} {'peak GB':>8}", flush=True)
micros = sorted({int(os.environ.get("MICRO_BS", 8)), 16})
for micro_bs in micros:
    for kda_impl in ("fla", "torch"):
        if kda_impl == "fla" and impl_req == "torch":
            continue
        for attn_impl in ("sdpa_mask", "sdpa", "manual"):
            for ckpt in (False, True):
                tok_s, peak = bench(micro_bs, kda_impl, attn_impl, ckpt)
                tag = "OOM" if tok_s is None else f"{tok_s:8.0f} {peak:8.2f}"
                print(f"{micro_bs:>5} {kda_impl:>6} {attn_impl:>10} "
                      f"{str(ckpt):>5} {tag}", flush=True)
