"""TinyBalls-V1 pretraining — run 2: KDA 3:1 hybrid.

Model: llama.HybridModel — KDA (Kimi Delta Attention) at layers 1-3, 5-7,
9-11 (1-indexed), full attention (GQA + QK-norm + learned sink + output
gate, NoPE) at 4, 8, 12. Everything else — data, schedule, precision
policy, checkpointing, hub sync — is byte-for-byte the run-1 baseline
recipe from llama/train.py, so the two runs differ only in the attention
stack. This file is a deliberate copy of llama/train.py with the model
construction swapped; run 1 is frozen mid-experiment and sharing a trainer
across model shapes was judged more regression risk than duplication.

Same token budget, same LR schedule (peak 6e-4, warmup 200, cosine to
0.1x, phase 2 = final 25%), same fp16 GradScaler policy. Known deviation:
the hybrid lands at ~124M params vs the baseline's 113M because Kimi-style
KDA carries low-rank gate paths a GQA layer doesn't have — see the
HybridModel docstring.

Env: identical to llama/train.py (TOK_DATA_DIR required; TOKENS_PER_STEP,
MICRO_BS, SEQ_LEN, PEAK_LR, WARMUP_STEPS, CKPT_EVERY, EVAL_EVERY,
MAX_HOURS, OUT_DIR, COMPILE, WANDB_*, HF_*), plus:
  CKPT_SAFETY   set to 0 to skip gradient checkpointing on the full-attention
                sublayers (faster if VRAM allows; default on for the score
                matrix, KDA layers never checkpointed)
  HF_CKPT_REPO  default <whoami>/tinyballs-v1-kda (separate repo from run 1)
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

from llama.data import PackedDataset, ValDataset
from llama.model import HybridModel

CKPT_REPO_SUFFIX = "tinyballs-v1-kda"
WANDB_RUN_NAME = "tinyballs-v1-kda"


# ---------------------------------------------------------------- groups

def param_groups(model, wd):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # tied head shows up twice
            continue
        seen.add(id(p))
        # embeddings, 1-D params (norm gains, A_log, dt_bias, sinks) no decay
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
        return f"{HfApi(token=token).whoami()['name']}/{CKPT_REPO_SUFFIX}"
    except Exception as e:
        print(f"[hf] no repo resolution ({e}); checkpoints stay local", flush=True)
        return ""


# ---------------------------------------------------------------- main

def main():
    data_dir = os.environ["TOK_DATA_DIR"]
    seq_len = int(os.environ.get("SEQ_LEN", 2048))
    micro_bs = int(os.environ.get("MICRO_BS", 8))
    tokens_per_step = int(os.environ.get("TOKENS_PER_STEP", 524288))
    peak_lr = float(os.environ.get("PEAK_LR", 6e-4))
    warmup = int(os.environ.get("WARMUP_STEPS", 200))
    ckpt_every = int(os.environ.get("CKPT_EVERY", 1000))
    eval_every = int(os.environ.get("EVAL_EVERY", 250))
    max_hours = float(os.environ.get("MAX_HOURS", 11))
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working")
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", "")
    hf_repo = os.environ.get("HF_CKPT_REPO") or (default_hf_repo(hf_token) if hf_token else "")
    started = time.time()

    # accum must be known before Accelerator construction: it owns the
    # gradient-accumulation sync (no_sync on non-final micros, one unscale +
    # step at the end). Without it, sync fires every micro — an all-reduce
    # and an optimizer step per micro-batch instead of per step.
    world = torch.cuda.device_count() if torch.cuda.is_available() else 1
    accum = tokens_per_step // (micro_bs * seq_len * world)
    if accum < 1:
        raise SystemExit(
            f"tokens_per_step {tokens_per_step} < one micro-batch "
            f"({micro_bs * seq_len * world}); raise TOKENS_PER_STEP or lower MICRO_BS")
    if accum * micro_bs * seq_len * world != tokens_per_step:
        print(f"note: tokens/step rounded to {accum * micro_bs * seq_len * world}", flush=True)

    use_cuda = torch.cuda.is_available()
    accelerator = Accelerator(
        mixed_precision="fp16" if use_cuda else None,
        gradient_accumulation_steps=accum,
        # opt-in: dynamo compile adds minutes of warmup and its own failure
        # modes; validated runs can flip COMPILE=1 in the launcher
        dynamo_backend="INDUCTOR" if os.environ.get("COMPILE") == "1" else "NO")

    vocab = 49154
    mpath = os.path.join(data_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            vocab = json.load(f).get("val", {}).get("vocab", vocab)
    # test knobs; production defaults match the hybrid spec
    dim = int(os.environ.get("D_MODEL", 768))
    n_layers = int(os.environ.get("N_LAYERS", 12))

    torch.manual_seed(1234)
    if use_cuda:
        from torch.backends.cuda import flash_sdp_enabled, math_sdp_enabled, mem_efficient_sdp_enabled
        print(f"sdpa flags: flash={flash_sdp_enabled()} "
              f"mem_efficient={mem_efficient_sdp_enabled()} math={math_sdp_enabled()}", flush=True)
        dev = torch.cuda.get_device_properties(0)
        print(f"gpu: {dev.name}, {dev.total_memory / 2**30:.1f} GB "
              f"x {world} ddp | micro={micro_bs} accum={accum}", flush=True)
    else:
        print(f"cpu-only smoke | micro={micro_bs} accum={accum}", flush=True)
    # production shape: 12 heads x 64 = dim 768; for small smoke tests
    # (D_MODEL/N_LAYERS env), shrink head_dim so heads * head_dim == dim
    n_heads = 12
    head_dim = 64 if dim == 768 else max(8, (dim // n_heads) * n_heads and dim // n_heads)
    if n_heads * head_dim != dim:
        n_heads = max(1, dim // 32)
        head_dim = 32
    model = HybridModel(vocab_size=vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                        n_kv_heads=max(1, n_heads // 3), head_dim=head_dim, ffn_dim=2048)
    if os.environ.get("CKPT_SAFETY", "1") != "0":
        for blk in model.blocks:
            blk.checkpoint_attn = not blk.is_kda
    print(f"params: {model.num_params()/1e6:.1f}M  vocab: {vocab}", flush=True)

    opt = torch.optim.AdamW(param_groups(model, wd=0.1), lr=peak_lr,
                            betas=(0.9, 0.95), fused=use_cuda)

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
                        name=os.environ.get("WANDB_NAME", WANDB_RUN_NAME),
                        config=dict(peak_lr=peak_lr, micro_bs=micro_bs, accum=accum,
                                    seq_len=seq_len, tokens_per_step=tokens_per_step,
                                    world_size=world, total_steps=total_steps,
                                    optimizer="adamw", betas=[0.9, 0.95],
                                    weight_decay=0.1,
                                    precision="fp16 (accelerate)",
                                    params_m=model.num_params() / 1e6, vocab=vocab,
                                    model="hybrid-kda",
                                    kda_layers=list(model.cfg["kda_layers"])))
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

        # skip/loss counters stay on-GPU during the accumulation loop; one
        # sync per step instead of one per micro (an .item() forces the host
        # to drain the launch queue before the next micro can be enqueued)
        dev = next(model.parameters()).device
        loss_t = torch.zeros(1, device=dev)
        skip_t = torch.zeros(1, device=dev, dtype=torch.int32)
        n_micro = 0
        for _ in range(accum):
            try:
                batch = next(it)
            except StopIteration:
                break
            with accelerator.accumulate(model):
                loss = model(batch[:, :-1], batch[:, 1:])
                skip_t += (~torch.isfinite(loss)).to(torch.int32)
                # non-finite forward (fp16 saturation). NaN * 0 is still
                # NaN, so rewriting the loss can't clean the graph; if the
                # activations themselves went inf, the backward pass
                # produces non-finite grads and the GradScaler skips the
                # optimizer step and backs off the scale. We must still
                # call backward every micro: skipping it would desync the
                # DDP allreduce when only some ranks see a NaN batch.
                loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            loss_t += loss.detach()
            n_micro += 1
            micros_done += 1
            tokens_seen += batch.numel() * world
        step_loss = accelerator.gather(loss_t).mean().item()
        n_skip = int(skip_t.item())
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
