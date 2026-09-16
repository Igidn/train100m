"""TinyBalls-V1 pretraining — LLaMA baseline experiment.

Two phases: phase 1 = broad mix (all *-p1-* shards), phase 2 = quality
anneal (all *-p2-* shards). One LR schedule spans both; phase 2 is the
final 25% of steps per the data spec. FP16 via Accelerate's fp16 mixed
precision (T4 has no bf16); the LM head always computes logits in fp32 —
under fp16 autocast a tied 49k-vocab head pushes logits past 65504 and
softmax turns inf into NaN. Lion optimizer — peak LR defaults to 3e-4,
the conservative end of Lion's 3-10x-below-AdamW rule (5e-4 diverged
before the fp32-head fix).
Bump via PEAK_LR in the launcher if the loss curve looks too flat.

DDP through `accelerate` (not torch.distributed directly): the launcher
starts this with `accelerate launch --num_processes <n_gpu>`. The token
budget per optimizer step is computed from TOKENS_PER_STEP and held
invariant across GPU counts — accumulation shrinks as world size grows.

Checkpoints go to /kaggle/working and are mirrored to the HF Hub so a
run survives Kaggle's 12h session limit and resumes on the next push.

Env:
  TOK_DATA_DIR      path to tok-mix-v1 root (required)
  TOKENS_PER_STEP   global token budget per optimizer step, default 524288
  MICRO_BS          micro-batch per GPU in sequences, default 4
  SEQ_LEN           default 2048
  PEAK_LR           default 5e-4
  WARMUP_STEPS      default 200
  CKPT_EVERY        save every N steps, default 1000
  EVAL_EVERY        val loss every N steps, default 250
  HF_CKPT_REPO      HF repo id for checkpoints; default <whoami>/tinyballs-v1
  WANDB_PROJECT     default tinyballs
  WANDB_NAME        default tinyballs-v1-llama
  MAX_HOURS         soft time budget: stop and save cleanly past it (11)
"""

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from accelerate import Accelerator

from src.data import PackedDataset, ValDataset
from src.model import LLaMA


# ---------------------------------------------------------------- Lion

class Lion(torch.optim.Optimizer):
    """Sign-based update; momentum uses half the memory of AdamW.

    Compatible with fp16 loss scaling: sign(s*g) == sign(g) for any
    positive scale s, so the scaler cancel-out leaves the direction
    untouched.
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

def save_ckpt(path, accelerator, model, opt, step, phase, micros_done, tokens_seen):
    if accelerator.is_main_process:
        tmp = path + ".tmp"
        torch.save(dict(
            model=accelerator.get_state_dict(model),
            opt=opt.state_dict(),
            step=step, phase=phase, micros_done=micros_done, tokens_seen=tokens_seen,
            rng=torch.get_rng_state(), np_rng=np.random.get_state(),
            py_rng=random.getstate(),
        ), tmp)
        os.replace(tmp, path)
    accelerator.wait_for_everyone()  # every rank sees the file before it's used


def load_ckpt_state(ck, model, opt):
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    torch.set_rng_state(ck["rng"])
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
        print(f"[hf] no repo resolution ({e}); checkpoints stay local", flush=True)
        return ""


# ---------------------------------------------------------------- main

def main():
    data_dir = os.environ["TOK_DATA_DIR"]
    seq_len = int(os.environ.get("SEQ_LEN", 2048))
    micro_bs = int(os.environ.get("MICRO_BS", 4))
    tokens_per_step = int(os.environ.get("TOKENS_PER_STEP", 524288))
    peak_lr = float(os.environ.get("PEAK_LR", 3e-4))
    warmup = int(os.environ.get("WARMUP_STEPS", 200))
    ckpt_every = int(os.environ.get("CKPT_EVERY", 1000))
    eval_every = int(os.environ.get("EVAL_EVERY", 250))
    max_hours = float(os.environ.get("MAX_HOURS", 11))
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    hf_repo = os.environ.get("HF_CKPT_REPO") or (default_hf_repo(hf_token) if hf_token else "")
    started = time.time()

    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else None)
    world = accelerator.num_processes

    accum = tokens_per_step // (micro_bs * seq_len * world)
    if accum < 1:
        raise SystemExit(
            f"tokens_per_step {tokens_per_step} < one micro-batch "
            f"({micro_bs * seq_len * world}); raise TOKENS_PER_STEP or lower MICRO_BS")
    if accum * micro_bs * seq_len * world != tokens_per_step:
        print(f"note: tokens/step rounded to {accum * micro_bs * seq_len * world}", flush=True)

    vocab = 49154
    mpath = os.path.join(data_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            vocab = json.load(f).get("val", {}).get("vocab", vocab)
    # test knobs; production defaults match the LLaMA baseline spec
    dim = int(os.environ.get("D_MODEL", 768))
    n_layers = int(os.environ.get("N_LAYERS", 12))

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(f"gpu: {dev.name}, {dev.total_memory / 2**30:.1f} GB "
              f"x {world} ddp | micro={micro_bs} accum={accum}", flush=True)
    else:
        print(f"cpu-only smoke | micro={micro_bs} accum={accum}", flush=True)
    model = LLaMA(vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=12,
                  n_kv_heads=4, head_dim=64, ffn_dim=2048)
    print(f"params: {model.num_params()/1e6:.1f}M  vocab: {vocab}", flush=True)

    opt = Lion(param_groups(model, wd=1.0), lr=peak_lr, betas=(0.9, 0.99))

    phase_ds = {1: PackedDataset(data_dir, 1, seq_len),
                2: PackedDataset(data_dir, 2, seq_len)}
    val_ds = ValDataset(data_dir, seq_len, n_chunks=256)
    print(f"p1: {phase_ds[1].tokens/1e6:.0f}M tok / {len(phase_ds[1])} chunks | "
          f"p2: {phase_ds[2].tokens/1e6:.0f}M tok / {len(phase_ds[2])} chunks | "
          f"val: {len(val_ds)} chunks", flush=True)

    micros = {p: len(ds) // micro_bs for p, ds in phase_ds.items()}
    steps = {p: micros[p] // (accum * world) for p in micros}
    total_steps = steps[1] + steps[2]
    phase2_start = steps[1]
    print(f"{micro_bs}x{accum}x{world} gpus -> {micro_bs * accum * world} seqs/step "
          f"({tokens_per_step // 1024}k tok/step); "
          f"steps {steps[1]}+{steps[2]}={total_steps}; peak lr {peak_lr}", flush=True)

    wandb = None
    if accelerator.is_main_process and os.environ.get("WANDB_API_KEY"):
        import wandb as _w
        _w.login(key=os.environ["WANDB_API_KEY"])
        wandb = _w.init(project=os.environ.get("WANDB_PROJECT", "tinyballs"),
                        name=os.environ.get("WANDB_NAME", "tinyballs-v1-llama"),
                        config=dict(peak_lr=peak_lr, micro_bs=micro_bs, accum=accum,
                                    seq_len=seq_len, tokens_per_step=tokens_per_step,
                                    world_size=world, total_steps=total_steps,
                                    optimizer="lion", betas=[0.9, 0.99],
                                    precision="fp16 (accelerate)",
                                    params_m=model.num_params() / 1e6, vocab=vocab))
    elif not os.environ.get("WANDB_API_KEY"):
        print("[wandb] no WANDB_API_KEY; stdout logging only", flush=True)

    # ---- resume: local checkpoint first, else latest checkpoint-N.pt on the Hub.
    # The file is on shared /kaggle/working, so every rank loads the same state.
    step, phase, micros_done, tokens_seen = 0, 1, 0, 0
    local_last = os.path.join(out_dir, "checkpoint_last.pt")
    ck_path = local_last
    if not os.path.exists(ck_path) and hf_repo and hf_token:
        if accelerator.is_main_process:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                names = api.list_repo_files(repo_id=hf_repo, repo_type="model")
                nums = sorted(int(f.split("-")[1].split(".")[0]) for f in names
                              if f.startswith("checkpoint-") and f != "checkpoint-final.pt")
                if nums:
                    p = api.hf_hub_download(repo_id=hf_repo, repo_type="model",
                                            filename=f"checkpoint-{nums[-1]}.pt")
                    import shutil
                    shutil.copyfile(p, ck_path)
                    print(f"fetched checkpoint-{nums[-1]} from HF", flush=True)
            except Exception as e:
                print(f"[hf] resume skipped: {e}", flush=True)
        accelerator.wait_for_everyone()
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        load_ckpt_state(ck, model, opt)
        step, phase = ck["step"], ck["phase"]
        micros_done, tokens_seen = ck["micros_done"], ck["tokens_seen"]
        print(f"resumed: step {step}/{total_steps}, phase {phase}", flush=True)
        del ck

    # move to GPUs + wrap DDP *after* the resume load, so the DDP init
    # broadcast carries the resumed weights to every rank
    model, opt = accelerator.prepare(model, opt)
    if accelerator.is_main_process:
        print(f"model device: {next(model.parameters()).device}", flush=True)

    # ---- data cursors; the shuffle is seeded per epoch, so resume = index offset
    def make_iter(phase, skip_micros=0):
        from torch.utils.data import DataLoader, Subset
        ds = phase_ds[phase]
        if skip_micros:
            ds = Subset(ds, range(skip_micros * micro_bs, len(ds)))
        loader = DataLoader(ds, batch_size=micro_bs, num_workers=2,
                            pin_memory=True, drop_last=True)
        return iter(accelerator.prepare(loader))

    if step == 0:
        it = make_iter(1)
    elif step < phase2_start:
        it = make_iter(1, micros_done)
    else:
        phase = 2
        it = make_iter(2, micros_done)

    # ---- eval (main process only; other ranks idle through it and rejoin)
    def evaluate():
        from torch.utils.data import DataLoader
        model.eval()
        dev = next(model.parameters()).device
        loss_sum, n = 0.0, 0
        loader = DataLoader(val_ds, batch_size=micro_bs, num_workers=2, drop_last=True)
        for b in loader:
            b = b.to(dev, non_blocking=True)
            with torch.no_grad():
                l = model(b[:, :-1], b[:, 1:])
            loss_sum += l.item() * b.shape[0]
            n += b.shape[0]
        model.train()
        return loss_sum / max(1, n)

    if step == 0:
        v = evaluate()
        if accelerator.is_main_process:
            print(f"initial val loss {v:.4f}", flush=True)
            if wandb:
                wandb.log(dict(val_loss=v), step=0)

    # ---- train
    model.train()
    log_every = 1
    next_log = step + 1
    skip_streak = 0
    t0, tokens_win = time.time(), 0
    stop_clean = False

    while step < total_steps and not stop_clean:
        if phase == 1 and step >= phase2_start:
            phase, micros_done = 2, 0
            it = make_iter(2)
            if accelerator.is_main_process:
                print("=== phase 2: quality anneal ===", flush=True)
        lr = lr_at(step, total_steps, peak_lr, warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        step_loss, n_micro, n_skip = 0.0, 0, 0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                break
            with accelerator.accumulate(model):
                loss = model(batch[:, :-1], batch[:, 1:])
                if not torch.isfinite(loss):
                    # non-finite forward (fp16 saturation). NaN * 0 is still
                    # NaN, so rewriting the loss can't clean the graph; if the
                    # activations themselves went inf, the backward pass
                    # produces non-finite grads and the GradScaler skips the
                    # optimizer step and backs off the scale. We must still
                    # call backward every micro: skipping it would desync the
                    # DDP allreduce when only some ranks see a NaN batch.
                    n_skip += 1
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step_loss += accelerator.gather(loss.detach()).mean().item()
            n_micro += 1
            micros_done += 1
            tokens_seen += batch.numel() * world
        step += 1
        tokens_win += micro_bs * accum * world * seq_len

        if n_skip == accum:
            skip_streak += 1
        else:
            skip_streak = 0
        if skip_streak >= 20:
            raise SystemExit(f"loss diverged: 20 consecutive fully-skipped steps "
                             f"ending at {step}; stopping to save the session quota")

        if step >= next_log and n_micro > 0:
            tps = tokens_win / (time.time() - t0)
            v = evaluate() if step % eval_every == 0 else None
            if accelerator.is_main_process:
                msg = (f"[{step}/{total_steps}] p{phase} loss {step_loss / n_micro:.4f} "
                       f"lr {lr:.2e} {tps / 1e3:.0f}k tok/s")
                if n_skip:
                    msg += f" ({n_skip} skip)"
                if v is not None:
                    msg += f" val {v:.4f}"
                print(msg, flush=True)
                rec = dict(step=step, loss=step_loss / n_micro, lr=lr,
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
            save_ckpt(local_last, accelerator, model, opt, step, phase,
                      micros_done, tokens_seen)
            if accelerator.is_main_process and hf_repo and hf_token:
                fname = f"checkpoint-{step}.pt"
                if hf_upload(local_last, hf_repo, fname, hf_token):
                    for f in os.listdir(out_dir):
                        if f.startswith("checkpoint-") and f != "checkpoint_last.pt":
                            os.remove(os.path.join(out_dir, f))
            accelerator.wait_for_everyone()

    # ---- final
    final_loss = evaluate()
    save_ckpt(local_last, accelerator, model, opt, step, phase,
              micros_done, tokens_seen)
    if accelerator.is_main_process:
        print(f"final val loss {final_loss:.4f}", flush=True)
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
    accelerator.wait_for_everyone()
    print("done", flush=True)


if __name__ == "__main__":
    main()
