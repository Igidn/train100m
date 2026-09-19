"""Thorough verification of checkpoint-final.pt (step 3827, phase 2, 2B tokens).

Structure/opt checks, val loss (first 8 per-chunk + full training-eval
population 0..255 + tail of the val file), EOS placement, ChatML exposure,
generation quality (greedy + top-p + prompts), induction/copy probes.

Prefill is vectorized through a KV-cache mirror of llama/model.py (validated
against the full forward); generation decodes incrementally from that cache.

Run:  python verify_final.py <ckpt> [quick|full|pi] [--stages 0,1,2,...]
"""
import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama.model import LLaMA, rotate_half

torch.set_num_threads(os.cpu_count() or 4)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

EOS, SEQ = 0, 2048
IM_START, IM_END = 1, 2

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default=os.path.expanduser("~/Downloads/checkpoint-final.pt"))
ap.add_argument("profile", nargs="?", default="quick", choices=["quick", "full", "pi"])
ap.add_argument("--stages", default="")
args = ap.parse_args()
STAGES = set(int(s) for s in args.stages.split(",") if s.strip()) if args.stages else None

def run_stage(n):
    return STAGES is None or n in STAGES

def p(*a, **kw):
    print(*a, flush=True)

def rss(tag=""):
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    kb = int(line.split()[1])
                    p(f"  [rss] {tag}: {kb/1024:.0f} MB", flush=True)
                    break
    except OSError:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = dict(quick=dict(val_extra=0, eos_docs=40, roll_docs=6, gen_steps=120,
                          probe_trials=24, chatml_gen=80),
               full=dict(val_extra=248, eos_docs=40, roll_docs=10, gen_steps=160,
                         probe_trials=48, chatml_gen=96),
               pi=dict(val_extra=248, eos_docs=40, roll_docs=6, gen_steps=160,
                       probe_trials=32, chatml_gen=96))[args.profile]

tok = None
try:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(HERE, "tokenizer.json"))
except ImportError:
    p("[warn] no tokenizers lib — text decode disabled")

p("== loading checkpoint ==")
t0 = time.time()
try:
    # mmap: tensor pages come from the file (file-backed, evictable) — on a
    # 4GB Pi the weights_only=False anon copy of a 1.4GB ckpt OOM-kills the run
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)
    res = dict(step=ck["step"], phase=ck["phase"], tokens_seen=ck["tokens_seen"])
    p(f"step {res['step']} phase {res['phase']} micros {ck['micros_done']} "
      f"tokens {res['tokens_seen']/1e9:.4f}B  (mmap load {time.time()-t0:.1f}s)")
except Exception as e:
    p(f"[mmap load failed: {e!r} — falling back to plain load]")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    res = dict(step=ck["step"], phase=ck["phase"], tokens_seen=ck["tokens_seen"])
    p(f"step {res['step']} phase {res['phase']} micros {ck['micros_done']} "
      f"tokens {res['tokens_seen']/1e9:.4f}B  (load {time.time()-t0:.1f}s)")
rss("after load")

model = LLaMA(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
              n_kv_heads=4, head_dim=64, ffn_dim=2048)
model.load_state_dict(ck["model"], strict=True)
model.eval()
rss("after model build")

META = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                    os.path.basename(args.ckpt).replace(".pt", "-meta.json"))

# ------------------------------------------------------------- stage 0: structure
if run_stage(0):
    p("\n== stage 0: structure / optimizer / finiteness ==", flush=True)
    p(f"state dict: strict load OK, params {model.num_params()/1e6:.1f}M (tied counted once)")
    p(f"tied head: lm_head.weight is tok_emb.weight -> "
      f"{model.lm_head.weight is model.tok_emb.weight}")

    nnan1 = nnan2 = 0
    nstates = steps_opt = lrs = None
    if "opt" in ck:                      # full checkpoint: live scan
        nnan1 = sum(1 for s in ck["opt"]["state"].values()
                    if "exp_avg" in s and not torch.isfinite(s["exp_avg"]).all())
        nnan2 = sum(1 for s in ck["opt"]["state"].values()
                    if "exp_avg_sq" in s and not torch.isfinite(s["exp_avg_sq"]).all())
        nstates = len(ck["opt"]["state"])
        steps_opt = [s.get("step") for s in ck["opt"]["state"].values()
                     if s.get("step") is not None]
        lrs = [g["lr"] for g in ck["opt"]["param_groups"]]
    elif os.path.exists(META):            # stripped file: checks were done at strip time
        with open(META) as f:
            meta = json.load(f)
        o = meta["opt"]
        nnan1 = o["nonfinite_exp_avg"]; nnan2 = o["nonfinite_exp_avg_sq"]
        nstates = o["nstates"]; steps_opt = [o["opt_step"]]; lrs = [o["lr"]]
        p(f"(opt checks from {os.path.basename(META)}; live opt state not in stripped file)")
    p(f"opt: {nstates} states, non-finite exp_avg {nnan1} exp_avg_sq {nnan2}, "
      f"opt-step {steps_opt[0] if steps_opt else '?'}, lr {lrs}")
    res["opt"] = dict(nstates=nstates, nonfinite=nnan1 + nnan2,
                      opt_step=steps_opt[0] if steps_opt else None, lr=lrs)

    wmax = max(v.abs().max().item() for v in ck["model"].values())
    n_bad = sum(1 for v in ck["model"].values() if not torch.isfinite(v).all())
    p(f"weight abs-max: {wmax:.3f}, non-finite model tensors: {n_bad}")
    res["weight_absmax"] = wmax
    n_rng = sum(k in ck for k in ("rng", "np_rng", "py_rng")) if "rng" in ck or "opt" in ck else None
    if n_rng is not None:
        p(f"rng keys present: {n_rng}/3")
    if "opt" in ck:
        rss("before ck clear")
        del ck["opt"]
        ck.clear()
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)   # return freed heap to the OS
        except Exception:
            pass
        rss("after ck clear")

    if tok is not None:
        ids = list(map(int, np.asarray(
            np.memmap(os.path.join(HERE, "val-000.bin"), dtype=np.uint16, mode="r")[0:64],
            dtype=np.int64)))
        rt = tok.encode(tok.decode(ids)).ids
        p(f"tokenizer round-trip (64 tok): {sum(a==b for a,b in zip(ids, rt))}/{len(ids)} match")
        for tid, name in [(EOS, "eos"), (IM_START, "im_start"), (IM_END, "im_end")]:
            p(f"  id {tid} -> {tok.decode([tid])!r}")

# ------------------------------------------------------- data + cached decoder
p("\n== data + KV-cache decoder: build + validate ==", flush=True)
tokens = np.memmap(os.path.join(HERE, "val-000.bin"), dtype=np.uint16, mode="r")
off = np.load(os.path.join(HERE, "val-000.offsets.npy"))
docs = [tokens[off[i]:off[i+1]] for i in range(len(off)-1)]
cand = [d for d in docs if len(d) >= 600][:PROFILE["eos_docs"]]
n_avail = os.path.getsize(os.path.join(HERE, "val-000.bin")) // 2 // SEQ

def _rms(x, w, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return y * w

class KVRunner:
    """KV-cache mirror of llama/model.py (fp32 CPU path).

    prefill_vec: whole prompt in one vectorized pass, cache filled.
    step(): one incremental token. logits_last(): head matmul.
    """
    def __init__(self, model, max_seq=2048):
        self.m = model
        cfg = model.cfg
        self.hd, self.nh, self.nkv = cfg["head_dim"], cfg["n_heads"], cfg["n_kv_heads"]
        self.eps = cfg["rms_eps"]
        self.n_rep = self.nh // self.nkv
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, self.hd, 2,
                                                        dtype=torch.float32) / self.hd))
        t = torch.arange(max_seq, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.cos = emb.cos()  # (max_seq, hd)
        self.sin = emb.sin()
        self.kbuf = torch.zeros(cfg["n_layers"], max_seq, self.nkv, self.hd)
        self.vbuf = torch.zeros(cfg["n_layers"], max_seq, self.nkv, self.hd)
        self.n = 0

    def reset(self):
        self.n = 0

    def prefill_vec(self, toks):
        """Vectorized prefill: one batched matmul per projection, cache filled.
        Returns (hidden (T, dim), logits_last (V,))."""
        m = self.m
        T = len(toks)
        x = m.tok_emb.weight[torch.tensor(toks, dtype=torch.long)]  # (T, dim)
        for li, blk in enumerate(m.blocks):
            h = _rms(x, blk.attn_norm.weight, self.eps)             # (T, dim)
            q = (h @ blk.attn.q_proj.weight.T).view(T, self.nh, self.hd)
            k = (h @ blk.attn.k_proj.weight.T).view(T, self.nkv, self.hd)
            v = (h @ blk.attn.v_proj.weight.T).view(T, self.nkv, self.hd)
            cos = self.cos[:T, None, :]                              # (T, 1, hd)
            sin = self.sin[:T, None, :]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            self.kbuf[li, :T] = k
            self.vbuf[li, :T] = v
            # attention via the same SDPA op llama/model.py uses (proven fast and
            # memory-lean on aarch64); 3D batched baddbmm/bmm dispatches to the
            # oneDNN ACL matmul backend here and allocates ~2GB workspace
            kx = self.kbuf[li, :T].repeat_interleave(self.n_rep, dim=1)  # (T, nh, hd)
            vx = self.vbuf[li, :T].repeat_interleave(self.n_rep, dim=1)
            q4 = q.permute(1, 0, 2).unsqueeze(0)     # (1, nh, T, hd)
            k4 = kx.permute(1, 0, 2).unsqueeze(0)
            v4 = vx.permute(1, 0, 2).unsqueeze(0)
            o = F.scaled_dot_product_attention(q4, k4, v4, is_causal=True)  # (1, nh, T, hd)
            o = o[0].permute(1, 0, 2).reshape(T, self.nh * self.hd)
            x = x + o @ blk.attn.o_proj.weight.T
            h2 = _rms(x, blk.mlp_norm.weight, self.eps)
            x = x + (F.silu(h2 @ blk.mlp.gate_proj.weight.T)
                     * (h2 @ blk.mlp.up_proj.weight.T)) @ blk.mlp.down_proj.weight.T
        xf = _rms(x, m.norm.weight, self.eps)
        logits_last = (xf[-1:] @ m.lm_head.weight.T)[0]              # (V,)
        self.n = T
        return xf, logits_last

    def _token(self, token):
        """one token through the stack; returns post-final-norm hidden (1,1,dim)"""
        m = self.m
        t = self.n
        cos, sin = self.cos[t], self.sin[t]   # (1, hd), broadcasts over heads
        x = m.tok_emb.weight[token:token+1, :]  # (1, 1, dim)
        for li, blk in enumerate(m.blocks):
            h = _rms(x, blk.attn_norm.weight, self.eps)
            q = (h @ blk.attn.q_proj.weight.T).view(self.nh, self.hd)
            k = (h @ blk.attn.k_proj.weight.T).view(self.nkv, self.hd)
            v = (h @ blk.attn.v_proj.weight.T).view(self.nkv, self.hd)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            self.kbuf[li, t:t+1] = k
            self.vbuf[li, t:t+1] = v
            T = t + 1
            kx = self.kbuf[li, :T].repeat_interleave(self.n_rep, dim=1)  # (T, nh, hd)
            vx = self.vbuf[li, :T].repeat_interleave(self.n_rep, dim=1)
            # broadcast-mul-reduce (no 3D matmul / einsum: ACL workspace balloon)
            scores = (kx * q[None]).sum(-1).transpose(0, 1) / math.sqrt(self.hd)  # (nh, T)
            att = torch.softmax(scores, dim=-1)                                  # (nh, T)
            o = (att.t()[:, :, None] * vx).sum(0)                                # (nh, hd)
            x = x + o.reshape(1, 1, self.nh * self.hd) @ blk.attn.o_proj.weight.T
            h2 = _rms(x, blk.mlp_norm.weight, self.eps)
            x = x + (F.silu(h2 @ blk.mlp.gate_proj.weight.T)
                     * (h2 @ blk.mlp.up_proj.weight.T)) @ blk.mlp.down_proj.weight.T
        return _rms(x, self.m.norm.weight, self.eps)

    def logits_last(self, out):
        return (out @ self.m.lm_head.weight.T)[0, 0]  # tied head

run = KVRunner(model)
p(f"built in {time.time()-t0:.1f}s")

# validate: vectorized prefill vs full forward, at several positions
doc = list(map(int, tokens[0:512]))
with torch.inference_mode():
    rss("before full fwd")
    lg_full = model(torch.tensor([doc]))[0]                     # (T, V)
    rss("after full fwd")
    t1 = time.time()
    xf, lg_cache = run.prefill_vec(doc)                         # xf: (T, dim)
    rss("after prefill_vec")
    dt_prefill = time.time() - t1
with torch.inference_mode():
    diff_last = (lg_full[-1] - lg_cache).abs().max().item()
    diff_mid = (lg_full[255] - (xf[255] @ model.lm_head.weight.T)).abs().max().item()
del lg_full, xf
with torch.inference_mode():
    nx = int(lg_cache.argmax())
    lg_step_full = model(torch.tensor([doc + [nx]]))[0, -1]
    lg_step_cache = run.logits_last(run._token(nx))
    rss("after decode step")
d2 = (lg_step_full - lg_step_cache).abs().max().item()
p(f"prefill_vec(512) {dt_prefill:.2f}s | logits max|diff| last {diff_last:.4f}, "
  f"mid(pos 255) {diff_mid:.4f} ({'OK' if max(diff_last, diff_mid) < 0.05 else 'WARN'})")
p(f"decode step logits max|diff|: {d2:.4f} ({'OK' if d2 < 0.05 else 'WARN'})")
res["cache_diff_prefill"] = [diff_last, diff_mid]
res["cache_diff_step"] = d2
del doc

def greedy_gen(runner, ctx, n_steps, stop=EOS):
    ctx = list(map(int, ctx))
    out = []
    runner.reset()
    _, lg = runner.prefill_vec(ctx)
    for i in range(n_steps):
        nx = int(lg.argmax())
        if nx == stop:
            return out, i, lg
        out.append(nx)
        lg = runner.logits_last(runner._token(nx))
        runner.n += 1
    return out, None, lg

def sample_gen(runner, ctx, n_steps, temp=0.9, top_p=0.95, stop=EOS, seed=0):
    g = torch.Generator().manual_seed(seed)
    ctx = list(map(int, ctx))
    out = []
    runner.reset()
    _, lg = runner.prefill_vec(ctx)
    for i in range(n_steps):
        pr = torch.softmax(lg.float() / temp, -1)
        if top_p < 1.0:
            sp, si = torch.sort(pr, descending=True)
            keep = (torch.cumsum(sp, -1) - sp) < top_p
            keep[0] = True
            pr = torch.zeros_like(pr).scatter_(0, si[keep], sp[keep])
            pr /= pr.sum()
        nx = int(torch.multinomial(pr, 1, generator=g))
        if nx == stop:
            return out, i, lg
        out.append(nx)
        lg = runner.logits_last(runner._token(nx))
        runner.n += 1
    return out, None, lg

def prefill_to(runner, ctx):
    """last-position logits for ctx (list)"""
    _, lg = runner.prefill_vec(list(map(int, ctx)))
    return lg

def rank_of(logits, tid):
    return (logits > logits[tid]).sum().item()

# ------------------------------------------------------------- stage 1: val loss
if run_stage(1):
    p(f"\n== stage 1: val loss (profile {args.profile}) ==", flush=True)
    per_chunk = []
    def eval_chunk(c):
        b = torch.from_numpy(np.asarray(tokens[c*SEQ:(c+1)*SEQ], dtype=np.int64))[None]
        with torch.inference_mode():
            return model(b[:, :-1], b[:, 1:]).item()

    tot = 0.0
    for c in range(8):
        t1 = time.time()
        l = eval_chunk(c)
        per_chunk.append(l)
        tot += l
        p(f"  chunk {c}: {l:.4f}  (mean8 {tot/(c+1):.4f})  [{time.time()-t1:.1f}s]")
    mean8 = tot / 8
    p(f"val loss first 8 chunks: {mean8:.4f}  ppl {math.exp(mean8):.2f}")
    res["val_loss_8"] = mean8
    res["val_chunks_8"] = per_chunk

    n_extra = PROFILE["val_extra"]
    if n_extra:
        bs = int(os.environ.get("VAL_BS", "4"))
        idxs = list(range(8, 8 + n_extra))
        t1 = time.time()
        tot2, n2 = sum(per_chunk), 8
        for bi in range(0, len(idxs), bs):
            grp = idxs[bi:bi+bs]
            b = torch.stack([torch.from_numpy(np.asarray(tokens[c*SEQ:(c+1)*SEQ],
                                                         dtype=np.int64)) for c in grp])
            with torch.inference_mode():
                l = model(b[:, :-1], b[:, 1:]).item()
            tot2 += l * b.shape[0]; n2 += b.shape[0]
            p(f"  chunks {grp[0]}-{grp[-1]}: batch loss {l:.4f}  (running {tot2/n2:.4f})  "
              f"[{n2}/256, {time.time()-t1:.0f}s]", flush=True)
        meanN = tot2 / n2
        p(f"val loss over full training-eval population (256 chunks, 0..255): "
          f"{meanN:.4f}  ppl {math.exp(meanN):.2f}")
        res["val_loss_256"] = meanN
        res["val_n_256"] = n2
        idxs_tail = [int(c) for c in np.unique(np.linspace(256, n_avail - 1, 24).astype(int))]
        tot3, n3 = 0.0, 0
        t1 = time.time()
        for bi in range(0, len(idxs_tail), bs):
            grp = idxs_tail[bi:bi+bs]
            b = torch.stack([torch.from_numpy(np.asarray(tokens[c*SEQ:(c+1)*SEQ],
                                                         dtype=np.int64)) for c in grp])
            with torch.inference_mode():
                l = model(b[:, :-1], b[:, 1:]).item()
            tot3 += l * b.shape[0]; n3 += b.shape[0]
            p(f"  chunks {grp[0]}-{grp[-1]}: batch loss {l:.4f}  (running {tot3/n3:.4f})  "
              f"[{n3}/{len(idxs_tail)}, {time.time()-t1:.0f}s]", flush=True)
        p(f"tail (chunks 256..{n_avail-1}) mean over {n3}: {tot3/n3:.4f} "
          f"ppl {math.exp(tot3/n3):.2f}")
        res["val_loss_tail24"] = tot3 / n3

# ------------------------------------------------------------- stage 2: EOS
if run_stage(2):
    p(f"\n== stage 2: EOS placement ({len(cand)} docs >= 600 tok) ==", flush=True)
    p(f"docs: {len(docs)} total, mean len {np.mean([len(d) for d in docs]):.0f}")

    ranks, end_p, endk = [], [], []
    t1 = time.time()
    with torch.inference_mode():
        for j, d in enumerate(cand):
            ctx = list(map(int, d[-513:-1]))
            xf, lg_last = run.prefill_vec(ctx)
            logits_last33 = (xf[-33:] @ run.m.lm_head.weight.T).float()  # (33, V)
            pr_eos = torch.softmax(logits_last33, -1)[:, EOS]            # pos 479..511
            ranks.append(rank_of(lg_last, EOS))
            end_p.append(float(torch.softmax(lg_last.float(), -1)[EOS]))
            endk.append(pr_eos.flip(0).numpy())   # endk[k-1] = p(eos) k before true end
            if (j + 1) % 10 == 0:
                p(f"  [{j+1}/{len(cand)}] {time.time()-t1:.0f}s", flush=True)
    ranks = np.array(ranks)
    endk = np.array(endk)
    p(f"eos rank at doc end: median {np.median(ranks):.0f} | "
      f"top1 {(ranks==0).sum()}/{len(cand)} top5 {(ranks<5).sum()}/{len(cand)} "
      f"top20 {(ranks<20).sum()}/{len(cand)}")
    p(f"eos prob at doc end: mean {np.mean(end_p):.3f} min {np.min(end_p):.4f} "
      f"max {np.max(end_p):.4f}")
    res["eos_end"] = dict(median_rank=float(np.median(ranks)),
                          top1=int((ranks==0).sum()), top5=int((ranks<5).sum()),
                          top20=int((ranks<20).sum()), mean_p=float(np.mean(end_p)))

    p("\n== eos rank at random mid-doc positions ==", flush=True)
    rng = np.random.default_rng(7)
    mid_r, mid_p = [], []
    with torch.inference_mode():
        for _ in range(len(cand)):
            d = cand[rng.integers(len(cand))]
            pos = int(rng.integers(512, len(d) - 100))
            lg = prefill_to(run, d[pos-512:pos])
            mid_r.append(rank_of(lg, EOS))
            mid_p.append(float(torch.softmax(lg.float(), -1)[EOS]))
    mid_r = np.array(mid_r)
    p(f"eos rank mid-doc: median {np.median(mid_r):.0f} top1 {(mid_r==0).sum()}/{len(mid_r)} "
      f"mean prob {np.mean(mid_p):.4f}")
    res["eos_mid"] = dict(median_rank=float(np.median(mid_r)),
                          top1=int((mid_r==0).sum()), mean_p=float(np.mean(mid_p)))

    p("\n== eos prob at end-k (same prefill pass as the end test) ==", flush=True)
    for k in (1, 2, 3, 5, 8, 12, 16, 24, 32):
        p(f"k={k:2d}: eos prob {endk[:, k-1].mean():.4f}")
    res["eos_endk"] = {str(k): float(endk[:, k-1].mean())
                       for k in (1, 2, 3, 5, 8, 12, 16, 24, 32)}

    p("\n== greedy rollout: doc body truncated at last k, steps to eos ==", flush=True)
    roll = {}
    for k in (1, 3, 10):
        hits, misses = [], 0
        with torch.inference_mode():
            for d in cand[:PROFILE["roll_docs"]]:
                body = list(map(int, d[:-1][:-k]))
                out, hit, _ = greedy_gen(run, body[-256:], k + 8)
                if hit is None:
                    misses += 1
                else:
                    hits.append(hit)
        p(f"k={k:2d}: eos {len(hits)}/{len(hits)+misses}, delays {hits}, no-eos {misses}")
        roll[str(k)] = dict(hits=len(hits), misses=misses, delays=hits)
    res["eos_rollout"] = roll

# ------------------------------------------------------------- stage 3: ChatML
if run_stage(3) and tok is not None:
    p("\n== stage 3: ChatML exposure & conditioning ==", flush=True)
    d0 = cand[0]
    with torch.inference_mode():
        lg_end = prefill_to(run, d0[-513:-1])
        lg_mid = prefill_to(run, d0[300:804])
        p_end, p_mid = torch.softmax(lg_end.float(), -1), torch.softmax(lg_mid.float(), -1)
        p(f"doc end : p(eos) {p_end[EOS]:.4f} p(im_end) {p_end[IM_END]:.8f} | "
          f"rank eos {rank_of(lg_end, EOS)} rank im_end {rank_of(lg_end, IM_END)}")
        p(f"doc mid : p(eos) {p_mid[EOS]:.4f} p(im_end) {p_mid[IM_END]:.8f} | "
          f"rank im_end {rank_of(lg_mid, IM_END)}")
        res["chatml"] = dict(p_eos_end=float(p_end[EOS]), p_im_end_end=float(p_end[IM_END]),
                             p_eos_mid=float(p_mid[EOS]), p_im_end_mid=float(p_mid[IM_END]))
        for label, txt in [("user", "<|im_start|>user\n"),
                           ("assistant", "<|im_start|>assistant\n")]:
            ctx = list(map(int, d0[:400])) + list(map(int, tok.encode(txt).ids))
            lg = prefill_to(run, ctx)
            tv, ti = torch.softmax(lg.float(), -1).topk(5)
            p(f"{label} after 400-tok doc prefix: top5 "
              f"{[(repr(tok.decode([i])), round(float(v), 4)) for v, i in zip(tv, ti)]}")
            p(f"   rank im_end {rank_of(lg, IM_END)}, p "
              f"{torch.softmax(lg.float(), -1)[IM_END]:.8f}")
        lg = prefill_to(run, tok.encode("<|im_start|>assistant\n").ids)
        tv, ti = torch.softmax(lg.float(), -1).topk(8)
        p(f"pure '<|im_start|>assistant\\n': top8 "
          f"{[(repr(tok.decode([i])), round(float(v), 4)) for v, i in zip(tv, ti)]}")
        ctx2 = list(map(int, tok.encode(
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
            "<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>\n"
            "<|im_start|>user\nAnd the capital of Germany?<|im_end|>\n"
            "<|im_start|>assistant\n").ids))
        out, hit, lg = greedy_gen(run, ctx2, PROFILE["chatml_gen"])
        stop = "<|im_end|>" if hit == IM_END else "<eos>" if hit == EOS else "no stop"
        p(f"2-turn ChatML gen ({len(out)} tok, {stop}): "
          f"{repr(tok.decode(out))[:220]}")
        pr = torch.softmax(lg.float(), -1)
        p(f"after answer: p(eos) {float(pr[EOS]):.4f} p(im_end) {float(pr[IM_END]):.8f}")
        res["chatml_2turn"] = dict(n=len(out), stop=hit,
                                   p_eos=float(pr[EOS]), p_im_end=float(pr[IM_END]))

# ------------------------------------------------------------- stage 4: generation
if run_stage(4) and tok is not None:
    p("\n== stage 4: free generation ==", flush=True)
    d = cand[4]
    with torch.inference_mode():
        def stats(out):
            ngrams = [tuple(out[i:i+4]) for i in range(len(out)-3)]
            rep4 = 1 - len(set(ngrams)) / max(1, len(ngrams))
            return len(set(out)) / max(1, len(out)), rep4
        out, hit, _ = greedy_gen(run, list(map(int, d[:64])), PROFILE["gen_steps"])
        dist, rep4 = stats(out)
        p(f"greedy 64-tok prefix: {len(out)} tok, eos "
          f"{'@'+str(hit) if hit is not None else 'no'}, distinct {dist:.2f}, rep4 {rep4:.2f}")
        p(f"  text: {repr(tok.decode(out))[:300]}")
        res["gen_greedy"] = dict(n=len(out), eos=hit, distinct=dist, rep4=rep4)
        for s in range(3):
            out, hit, _ = sample_gen(run, list(map(int, d[:64])), PROFILE["gen_steps"],
                                     temp=0.9, top_p=0.95, seed=1234 + s)
            dist, rep4 = stats(out)
            p(f"top-p 0.95 t0.9 sample {s}: {len(out)} tok, eos "
              f"{'@'+str(hit) if hit is not None else 'no'}, distinct {dist:.2f}, rep4 {rep4:.2f}")
            p(f"  text: {repr(tok.decode(out))[:300]}")
            res[f"gen_sample{s}"] = dict(n=len(out), eos=hit, distinct=dist, rep4=rep4)
        for prompt in ["The capital of France is", "Once upon a time",
                       "def quicksort(arr):", "The water cycle consists of"]:
            out, hit, _ = greedy_gen(run, list(map(int, tok.encode(prompt).ids)), 60)
            p(f"completion {prompt!r}: {repr(tok.decode(out))[:200]}")

# ------------------------------------------------------------- stage 5: probes
if run_stage(5):
    p("\n== stage 5: induction / copy probes ==", flush=True)
    n_trials = PROFILE["probe_trials"]
    rng = np.random.default_rng(11)
    def probe(alen, ks):
        hits = {k: 0 for k in ks}
        with torch.inference_mode():
            for _ in range(n_trials):
                base = [int(t) for t in rng.integers(1000, 48000, alen)]
                for k in ks:
                    lg = prefill_to(run, base + base[:k])
                    if int(lg.argmax()) == base[k]:
                        hits[k] += 1
        return {str(k): hits[k] / n_trials for k in ks}
    for alen, ks in [(16, [1, 2, 3, 5, 8]), (32, [1, 2, 3, 5, 8]), (64, [1, 2, 3, 5])]:
        acc = probe(alen, ks)
        p(f"induction probe alphabet {alen}: " + " ".join(f"k={k}:{acc[str(k)]:.0%}" for k in ks))
        res[f"induction_{alen}"] = acc
    with torch.inference_mode():
        ctx = [int(t) for t in np.random.default_rng(5).integers(1000, 48000, 16)]
        out, _, _ = greedy_gen(run, ctx, 32)
        dist = len(set(out)) / max(1, len(out))
        p(f"random-16 continuation: distinct {dist:.2f}, "
          f"{repr(tok.decode(out) if tok else out)[:120]}")
        res["rand16_distinct"] = dist

# ------------------------------------------------------------- stage 6: logit health
if run_stage(6):
    p("\n== stage 6: logit health & finiteness sweep ==", flush=True)
    with torch.inference_mode():
        lg = prefill_to(run, cand[0][-513:-1])
        p(f"logit range: min {lg.min().item():.1f} max {lg.max().item():.1f} "
          f"finite {torch.isfinite(lg).all().item()}")
        pr = torch.softmax(lg.float(), -1)
        tv, ti = pr.topk(5)
        p(f"top5 prob {tv.tolist()}")
        if tok is not None:
            p(f"top5 tokens {[repr(tok.decode([i])) for i in ti.tolist()]}")
        r2 = np.random.default_rng(3)
        bad = 0
        for _ in range(20):
            lg2 = prefill_to(run, [int(t) for t in r2.integers(0, 49154, 256)])
            if not torch.isfinite(lg2).all():
                bad += 1
        p(f"noise-sweep forwards non-finite: {bad}/20")
        res["logit_health"] = dict(min=float(lg.min()), max=float(lg.max()),
                                   nonfinite_sweep=bad)

p("\n== done ==", flush=True)
with open(os.path.join(HERE, f"verify_final_{args.profile}.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
p("results json written")
