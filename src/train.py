"""TinyBalls-V1 pretraining — LLaMA baseline experiment.

Two phases: phase 1 = broad mix (all *-p1-* shards), phase 2 = quality
anneal (all *-p2-* shards). One LR schedule spans both; phase 2 is the
final 25% of steps per the data spec. FP16 AMP + GradScaler (T4 has no
bf16), Lion optimizer — note Lion conventionally runs 3-10x smaller LR
than AdamW for the same task, so the default peak LR here is ~5e-4
(AdamW-equivalent ~3e-3 / 6). Bump via PEAK_LR in the launcher if the
loss curve says so.

Everything is driven by env vars; the launcher (not this file) owns the
dataset path and secret handling. Checkpoints go to /kaggle/working and
are mirrored to the HF Hub so a run survives Kaggle's 12h session limit
and resumes on the next push.

Env:
  TOK_DATA_DIR      path to tok-mix-v1 root (required)
  PEAK_LR           default 5e-4
  MICRO_BS          micro-batch size in sequences, default 8
  ACCUM             grad accumulation micro-steps, default 32
  SEQ_LEN           default 2048
  CKPT_EVERY        save every N steps, default 1000
  EVAL_EVERY        val loss every N steps, default 250
  HF_CKPT_REPO      HF repo id for checkpoints; default <whoami>/tinyballs-v1
  WANDB_PROJECT     default tinyballs
  WANDB_NAME        default tinyballs-v1-llama
  WARMUP_STEPS      default 200
  MAX_HOURS         soft time budget: stop and save cleanly past it (11)
"""

import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import PackedDataset, ValDataset
from src.model import LLaMA


# ---------------------------------------------------------------- Lion

class Lion(torch.optim.Optimizer):
    """Sign-based update; momentum uses half the memory of AdamW.

    Compatible with fp16 GradScaler: sign(s*g) == sign(g) for any
    positive scale s, so the loss-scale cancel-out leaves Lion's
    direction untouched.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                upd = m.mul(b1).add_(g, alpha=1 - b1).sign_()
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(upd, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)


def param_groups(model, wd):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # tied head shows up twice
            continue
        seen.add(id(p))
        # embeddings and 1-D params (norm gains) get no decay
        (no_decay if ("emb" in name or p.ndim < 2) else decay).append(p)
    return [dict(params=decay, weight_decay=wd),
            dict(params=no_decay, weight_decay=0.0)]


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------- ckpt io

def save_ckpt(path, model, opt, scaler, step, phase, micros_done, tokens_seen):
    tmp = path + ".tmp"
    torch.save(dict(
        model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
        step=step, phase=phase, micros_done=micros_done, tokens_seen=tokens_seen,
        rng=torch.get_rng_state(), np_rng=np.random.get_state(), py_rng=random.getstate(),
    ), tmp)
    os.replace(tmp, path)


def load_ckpt_state(ck, model, opt, scaler):
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    torch.set_rng_state(ck["rng"].cpu())
    np.random.set_state(ck["np_rng"])
    random.setstate(tuple(ck["py_rng"]))


def hf_upload(local_path, repo, fname, token):
    try:
        from huggingface_hub import HfApi
        HfApi(token=token).upload_file(
            path_or_fileobj=local_path, path_in_repo=fname,
            repo_id=repo, repo_type="model")
        return True
    except Exception as e:  # upload failures must never kill the run
        print(f"[hf] upload failed: {e}", flush=True)
        return False


def default_hf_repo(token):
    try:
        from huggingface_hub import HfApi
        return f"{HfApi(token=token).whoami()['name']}/tinyballs-v1"
    except Exception as e:
        print(f"[hf] repo resolution failed ({e}); checkpoints stay local", flush=True)
        return ""


# ---------------------------------------------------------------- main

def main():
    data_dir = os.environ["TOK_DATA_DIR"]
    seq_len = int(os.environ.get("SEQ_LEN", 2048))
    micro_bs = int(os.environ.get("MICRO_BS", 8))
    accum = int(os.environ.get("ACCUM", 32))
    peak_lr = float(os.environ.get("PEAK_LR", 5e-4))
    warmup = int(os.environ.get("WARMUP_STEPS", 200))
    ckpt_every = int(os.environ.get("CKPT_EVERY", 1000))
    eval_every = int(os.environ.get("EVAL_EVERY", 250))
    max_hours = float(os.environ.get("MAX_HOURS", 11))
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    hf_repo = os.environ.get("HF_CKPT_REPO") or (default_hf_repo(hf_token) if hf_token else "")
    started = time.time()

    vocab = 49154
    mpath = os.path.join(data_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            vocab = json.load(f).get("val", {}).get("vocab", vocab)

    torch.manual_seed(1234)
    model = LLaMA(vocab_size=vocab, dim=768, n_layers=12, n_heads=12,
                  n_kv_heads=4, head_dim=64, ffn_dim=2048).cuda()
    print(f"params: {model.num_params()/1e6:.1f}M  vocab: {vocab}", flush=True)

    opt = Lion(param_groups(model, wd=1.0), lr=peak_lr, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler("cuda")

    phase_ds = {1: PackedDataset(data_dir, 1, seq_len),
                2: PackedDataset(data_dir, 2, seq_len)}
    val_ds = ValDataset(data_dir, seq_len, n_chunks=256)
    print(f"p1: {phase_ds[1].tokens/1e6:.0f}M tok / {len(phase_ds[1])} chunks | "
          f"p2: {phase_ds[2].tokens/1e6:.0f}M tok / {len(phase_ds[2])} chunks | "
          f"val: {len(val_ds)} chunks", flush=True)

    seqs_per_step = micro_bs * accum
    micros = {p: len(ds) // micro_bs for p, ds in phase_ds.items()}
    steps = {p: micros[p] // accum for p in micros}
    total_steps = steps[1] + steps[2]
    phase2_start = steps[1]
    print(f"micro={micro_bs} accum={accum} -> {seqs_per_step} seqs/step "
          f"({seqs_per_step * seq_len / 1024:.0f}k tok/step); "
          f"steps {steps[1]}+{steps[2]}={total_steps}; peak lr {peak_lr}", flush=True)

    wandb = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb as _w
        _w.login(key=os.environ["WANDB_API_KEY"])
        wandb = _w.init(project=os.environ.get("WANDB_PROJECT", "tinyballs"),
                        name=os.environ.get("WANDB_NAME", "tinyballs-v1-llama"),
                        config=dict(peak_lr=peak_lr, micro_bs=micro_bs, accum=accum,
                                    seq_len=seq_len, seqs_per_step=seqs_per_step,
                                    total_steps=total_steps, optimizer="lion",
                                    betas=[0.9, 0.99], precision="fp16+scaler",
                                    params_m=model.num_params() / 1e6, vocab=vocab))
    else:
        print("[wandb] no WANDB_API_KEY; stdout logging only", flush=True)

    # ---- resume: local checkpoint first, else latest checkpoint-N.pt on the Hub
    step, phase, micros_done, tokens_seen = 0, 1, 0, 0
    local_last = os.path.join(out_dir, "checkpoint_last.pt")
    ck = None
    if os.path.exists(local_last):
        ck = torch.load(local_last, map_location="cuda", weights_only=False)
        print(f"local checkpoint: step {ck['step']}, phase {ck['phase']}", flush=True)
    elif hf_repo and hf_token:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_token)
            names = api.list_repo_files(repo_id=hf_repo, repo_type="model")
            nums = sorted(int(f.split("-")[1].split(".")[0]) for f in names
                          if f.startswith("checkpoint-") and f != "checkpoint-final.pt")
            if nums:
                last = nums[-1]
                p = api.hf_hub_download(repo_id=hf_repo, repo_type="model",
                                        filename=f"checkpoint-{last}.pt")
                ck = torch.load(p, map_location="cuda", weights_only=False)
                print(f"resuming from HF checkpoint-{last}", flush=True)
        except Exception as e:  # no usable Hub state -> fresh run
            print(f"[hf] resume skipped: {e}", flush=True)
    if ck is not None:
        load_ckpt_state(ck, model, opt, scaler)
        step, phase = ck["step"], ck["phase"]
        micros_done, tokens_seen = ck["micros_done"], ck["tokens_seen"]
        print(f"resumed: step {step}/{total_steps}, phase {phase}", flush=True)

    # ---- eval
    def evaluate():
        model.eval()
        loss_sum, n = 0.0, 0
        loader = DataLoader(val_ds, batch_size=micro_bs, num_workers=2, drop_last=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in loader:
                b = b.cuda(non_blocking=True)
                l = model(b[:, :-1], b[:, 1:])
                loss_sum += l.item() * b.shape[0]
                n += b.shape[0]
        model.train()
        return loss_sum / max(1, n)

    if step == 0:
        v = evaluate()
        print(f"initial val loss {v:.4f}", flush=True)
        if wandb:
            wandb.log(dict(val_loss=v), step=0)

    # ---- train
    model.train()

    from torch.utils.data import Subset

    def make_iter(phase, skip_micros=0):
        ds = phase_ds[phase]
        if skip_micros:
            ds = Subset(ds, range(skip_micros * micro_bs, len(ds)))
        loader = DataLoader(ds, batch_size=micro_bs, num_workers=2,
                            pin_memory=True, drop_last=True, persistent_workers=True)
        return iter(loader)

    # data cursor: the dataset order is deterministic, so resuming is just
    # offsetting past already-consumed micro-batches
    if step == 0:
        it = make_iter(1)
    elif step < phase2_start:
        it = make_iter(1, micros_done)
    else:
        phase = 2
        it = make_iter(2, micros_done)

    log_every = 10
    next_log = step + 1
    t0, tokens_win = time.time(), 0
    stop_clean = False

    while step < total_steps and not stop_clean:
        if phase == 1 and step >= phase2_start:
            phase, micros_done = 2, 0
            it = make_iter(2)
            print("=== phase 2: quality anneal ===", flush=True)
        lr = lr_at(step, total_steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = batch.cuda(non_blocking=True)
            loss = model(batch[:, :-1], batch[:, 1:])
            scaler.scale(loss / accum).backward()
            step_loss += loss.item()
            micros_done += 1
            tokens_seen += batch.numel()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1
        tokens_win += seqs_per_step * seq_len

        if step >= next_log:
            tps = tokens_win / (time.time() - t0)
            v = evaluate() if step % eval_every == 0 else None
            msg = (f"[{step}/{total_steps}] p{phase} loss {step_loss / accum:.4f} "
                   f"lr {lr:.2e} {tps / 1e3:.0f}k tok/s")
            if v is not None:
                msg += f" val {v:.4f}"
            print(msg, flush=True)
            rec = dict(step=step, loss=step_loss / accum, lr=lr,
                       tokens_per_sec=tps, phase=phase, tokens_seen=tokens_seen)
            if v is not None:
                rec["val_loss"] = v
            if wandb:
                wandb.log(rec, step=step)
            t0, tokens_win = time.time(), 0
            next_log = step + log_every
            if (time.time() - started) / 3600 > max_hours:
                stop_clean = True

        if step % ckpt_every == 0 or stop_clean or step == total_steps:
            save_ckpt(local_last, model, opt, scaler, step, phase, micros_done, tokens_seen)
            if hf_repo and hf_token:
                fname = f"checkpoint-{step}.pt"
                if hf_upload(local_last, hf_repo, fname, hf_token):
                    for f in os.listdir(out_dir):
                        if f.startswith("checkpoint-") and f != "checkpoint_last.pt":
                            os.remove(os.path.join(out_dir, f))

    # ---- final
    final_loss = evaluate()
    print(f"final val loss {final_loss:.4f}", flush=True)
    save_ckpt(local_last, model, opt, scaler, step, phase, micros_done, tokens_seen)
    if hf_repo and hf_token:
        hf_upload(local_last, hf_repo, "checkpoint-final.pt", hf_token)
        try:
            p = os.path.join(out_dir, "final_config.json")
            with open(p, "w") as f:
                json.dump(dict(model.cfg, params_m=model.num_params() / 1e6,
                               final_val_loss=final_loss, steps=step,
                               tokens=tokens_seen), f, indent=2)
            hf_upload(p, hf_repo, "final_config.json", hf_token)
        except Exception as e:
            print(f"[hf] final config upload failed: {e}", flush=True)
    if wandb:
        wandb.log(dict(final_val_loss=final_loss), step=step)
        wandb.finish()
    print("done", flush=True)


if __name__ == "__main__":
    main()
