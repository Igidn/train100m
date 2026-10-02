#!/usr/bin/env python
"""TinyBalls-V1 tool-call repair SFT — packed ChatML tuning on repair-tok-v1.

Continues the finished SFT run (igidn/tinyballs-v1 `SFT/final.pt`, rope theta
500k) with the targeted repair pass from sft/TOOL_FIX_PLAN.md 5: schema
binding to the tools declared in *this* prompt, error recovery, abstention,
and grounded final answers. This is a specialised copy of sft/train_sft.py —
same trainer, same packing/loss-mask semantics (llama/sft_data.py), same
checkpoint/resume machinery; the SFT script itself is left untouched.

Default schedule (single phase; every knob is env-tunable):

  repair   seq 8192, theta 500k, short+mid buckets, 2 epochs, LR 2e-5 with a
           30-step warmup and cosine decay to floor 0.1, grad ckpt auto
  long     seq 32768, theta 500k, long bucket, 1 epoch, LR 1e-5 — opt-in via
           PHASES=repair,long, only worth it if 32k tool use is a target (the
           long bucket holds just 186 docs / 2.5M tokens)

The repair mix carries its own anti-regression replay (3,500 general instruct
rows, ~5% of rows / ~11% of tokens), so REPLAY_FRAC defaults to 0.0. The
tok-mix-v1 raw-pretrain replay used by SFT is still supported: mount the
dataset and set REPLAY_FRAC>0 to mix *-p2-* chunks back in.

One caveat on the opt-in long phase: `done` is only published for the
canonical PHASES=repair schedule, and a completed repair-only run makes later
pushes exit early — run PHASES=repair,long in one schedule (or clear
REPAIR/state.json first) if 32k is wanted.

Precision: T4 = sm75, fp16 + Accelerate GradScaler (what the pretrain and SFT
runs used). Sized for 16GB T4s: micro-batch 1 sequence/GPU, block gradient
checkpointing above 8k context, chunked fp16 cross-entropy with masked fp32
reduction (a full 32k x 49k logit tensor never lands in one piece).

Resumability: identical to the SFT trainer — training is a pure function of
the global step (data plans are seeded; no loader cursors to save), and
checkpoints carry model + fp32 AdamW + GradScaler + RNG state. State lives in
the base repo under REPAIR/ so a session that dies at Kaggle's 12h wall is
resumed by pushing the kernel again:

  REPAIR/state.json            tiny progress record; read first on resume
  REPAIR/checkpoint-<step>.pt  model + optimizer + scaler (newest 2 kept)
  REPAIR/final.pt              model + full cfg, this repo's own format
  REPAIR/model.safetensors     HF LlamaForCausalLM-compatible export (final)
  REPAIR/config.json           HF arch config (rope_theta of the last phase)
  REPAIR/{tokenizer.json,tokenizer_config.json,generation_config.json,README.md}

A 1.4GB checkpoint upload runs on a background thread so the GPUs never wait
on the Hub; uploads retry a few times before they are dropped. state.json is
only published for steps whose checkpoint was written — a MAX_HOURS stop
between checkpoint intervals keeps the previous step — and resume falls back
to the newest checkpoint actually on the Hub when the exact one is missing,
so a wall stop never becomes a restart from scratch.

Env:
  REPAIR_DATA_DIR   repair-tok-v1 root (required; short/ mid/ long/ manifest)
  TOK_DATA_DIR      tok-mix-v1 root for raw replay (only if REPLAY_FRAC>0)
  PHASES            comma list, default "repair" (selects from repair,long)
  SEQ_* THETA_* EPOCHS_* LR_* WARMUP_* LR_FLOOR_*
                    per-phase schedule overrides (defaults above)
  REPLAY_FRAC       replay row share per phase, default 0.0
  TOKENS_PER_STEP   global token budget per optimizer step, default 65536
  MICRO_BS          sequences per GPU per forward, default 1
  CE_CHUNK          vocab chunk size for the chunked CE, default 2048
  GRAD_CKPT         "auto" (default: on when seq > 8192), "1", "0"
  MAX_POS           context advertised in the exported HF config, default
                    32768 (the SFT-trained context; repair runs at 8k)
  CKPT_EVERY        checkpoint/upload interval in steps, default 250
  EVAL_EVERY        val-loss interval in steps, default 250
  MAX_HOURS         soft stop, save + exit cleanly past it, default 11
  MAX_STEPS         cap on *new* steps this session (smoke tests), default 0
  OUT_DIR           working dir, default /kaggle/working
  BASE_REPO / BASE_CKPT_FILE / BASE_CKPT_PATH   base weights source
                    default igidn/tinyballs-v1 + SFT/final.pt
  HF_CKPT_REPO / HF_CKPT_DIR                    output repo, default
                    igidn/tinyballs-v1 + REPAIR
  D_MODEL N_LAYERS N_HEADS N_KV_HEADS HEAD_DIM FFN_DIM VOCAB
                    arch overrides; defaults are the run-1 LLaMA spec, which
                    the base state dict must match exactly (strict load)
  SEED              default 1234
  WANDB_PROJECT WANDB_NAME WANDB_API_KEY
"""

import glob
import io
import json
import math
import os
import queue
import random
import shutil
import sys
import threading
import time
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from accelerate import Accelerator

from llama.model import LLaMA
from llama.sft_data import ChunkPool, SFTPhaseData

DEFAULT_PHASES = [
    # name, seq, theta, buckets, epochs, lr, warmup, floor, replay, cpt_tokens
    ("repair", 8192,  500000.0, "short,mid", 2, 2e-5, 30, 0.1, 0.00, 0),
    ("long",   32768, 500000.0, "long",      1, 1e-5, 10, 0.1, 0.00, 0),
]


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def param_groups(model, wd):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:  # tied head shows up twice
            continue
        seen.add(id(p))
        # embeddings, 1-D params (norm gains) -> no decay
        (no_decay if ("emb" in name or p.ndim < 2) else decay).append(p)
    return [dict(params=decay, weight_decay=wd),
            dict(params=no_decay, weight_decay=0.0)]


def lr_at(step, total, peak, warmup, floor):
    """Linear warmup, then cosine to floor*peak (floor=0.1 is the pretrain shape)."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t)))


def phase_specs():
    """Phase table from env; PHASES selects/orders a subset."""
    want = [p.strip() for p in os.environ.get("PHASES", "repair").split(",")
            if p.strip()]
    specs = []
    for name, seq, theta, buckets, epochs, lr, warmup, floor, replay, cpt in DEFAULT_PHASES:
        if name not in want:
            continue
        up = name.upper()
        specs.append(dict(
            name=name,
            seq=env_int(f"SEQ_{up}", seq),
            theta=env_float(f"THETA_{up}", theta),
            buckets=[b for b in os.environ.get(f"BUCKETS_{up}", buckets).split(",") if b],
            epochs=env_int(f"EPOCHS_{up}", epochs),
            lr=env_float(f"LR_{up}", lr),
            warmup=env_int(f"WARMUP_{up}", warmup),
            floor=env_float(f"LR_FLOOR_{up}", floor),
            replay=env_float("REPLAY_FRAC", replay),
            cpt_tokens=env_int("CPT_TOKENS_LONG", cpt) if name == "long" else 0,
        ))
    return specs


# ---------------------------------------------------------------- checkpoint

def move_state_dict(sd):
    return {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}


def save_ckpt(path, accelerator, model, opt, extra):
    """Single-file checkpoint on shared /kaggle/working; every rank waits."""
    if accelerator.is_main_process:
        payload = dict(model=move_state_dict(accelerator.get_state_dict(model)),
                       opt=opt.state_dict(), **extra)
        if accelerator.scaler is not None:
            payload["scaler"] = accelerator.scaler.state_dict()
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
    accelerator.wait_for_everyone()


def load_ckpt(path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=True)
    if opt is not None and "opt" in ck:
        opt.load_state_dict(ck["opt"])
    if scaler is not None and "scaler" in ck:
        scaler.load_state_dict(ck["scaler"])
    for key in ("rng", "np_rng", "py_rng"):
        if key not in ck:
            continue
        if key == "rng":
            torch.set_rng_state(ck[key])
        elif key == "np_rng":
            np.random.set_state(ck[key])
        else:
            random.setstate(tuple(ck[key]))
    return ck


# ---------------------------------------------------------------- hf io

class HFUploader(threading.Thread):
    """Background single-file uploader; the trainer never blocks on the Hub.

    Items: ("ckpt", path, step) uploads REPAIR/checkpoint-<step>.pt; the
    ("state", dict) queued after it records that step; ("clean", step) then
    prunes numbered checkpoints beyond the newest `keep`. Transient failures
    are retried; anything still failing is printed and swallowed — the Hub
    must never kill a run.
    """

    def __init__(self, repo, token, folder, keep=2):
        super().__init__(daemon=True)
        self.repo, self.token, self.folder = repo, token, folder
        self.keep = keep
        self.q = queue.Queue()
        self.last_step = -1  # last checkpoint step confirmed on the Hub

    def submit(self, item):
        self.q.put(item)

    def _upload(self, api, **kwargs):
        """api.upload_file with a few retries; raises after the last attempt."""
        for attempt in range(3):
            try:
                api.upload_file(**kwargs)
                return
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"[hf] upload failed ({e}); retrying", flush=True)
                time.sleep(5 * (attempt + 1))

    def run(self):
        from huggingface_hub import HfApi
        api = HfApi(token=self.token)
        while True:
            item = self.q.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "ckpt":
                    _, path, step = item
                    self._upload(api, path_or_fileobj=path,
                                 path_in_repo=f"{self.folder}/checkpoint-{step}.pt",
                                 repo_id=self.repo, repo_type="model")
                    self.last_step = step
                    print(f"[hf] uploaded checkpoint-{step}.pt", flush=True)
                elif kind == "state":
                    self._upload(api,
                                 path_or_fileobj=io.BytesIO(json.dumps(item[1], indent=1).encode()),
                                 path_in_repo=f"{self.folder}/state.json",
                                 repo_id=self.repo, repo_type="model")
                elif kind == "file":
                    _, path, name = item
                    self._upload(api, path_or_fileobj=path,
                                 path_in_repo=f"{self.folder}/{name}",
                                 repo_id=self.repo, repo_type="model")
                    print(f"[hf] uploaded {name}", flush=True)
                elif kind == "clean":
                    _, step = item
                    names = api.list_repo_files(repo_id=self.repo, repo_type="model")
                    nums = sorted(int(n.split("/")[-1].split("-")[1].split(".")[0])
                                  for n in names
                                  if n.startswith(f"{self.folder}/checkpoint-")
                                  and n.endswith(".pt"))
                    for n in nums[:-self.keep]:
                        api.delete_file(path_in_repo=f"{self.folder}/checkpoint-{n}.pt",
                                        repo_id=self.repo, repo_type="model")
                        print(f"[hf] pruned checkpoint-{n}.pt", flush=True)
            except Exception as e:
                print(f"[hf] uploader error on {item[0]}: {e}", flush=True)

    def drain(self, timeout=1800):
        self.q.put(None)
        self.join(timeout=timeout)
        if self.is_alive():
            print("[hf] uploader still busy after drain timeout", flush=True)


def fetch_remote_state(env, out_dir):
    if not env["hf_repo"] or not env["hf_token"]:
        return None
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=env["hf_repo"],
                            filename=f"{env['hf_dir']}/state.json",
                            token=env["hf_token"],
                            local_dir=os.path.join(out_dir, "hf_resume"))
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"[hf] no resume state ({type(e).__name__}: {e})", flush=True)
        return None


def remote_ckpt_steps(env):
    """Steps with REPAIR/checkpoint-<step>.pt on the Hub, ascending ([] on failure)."""
    if not env["hf_repo"] or not env["hf_token"]:
        return []
    try:
        from huggingface_hub import HfApi
        names = HfApi(token=env["hf_token"]).list_repo_files(
            repo_id=env["hf_repo"], repo_type="model")
    except Exception as e:
        print(f"[hf] checkpoint listing failed: {e}", flush=True)
        return []
    prefix = f"{env['hf_dir']}/checkpoint-"
    steps = []
    for n in names:
        if n.startswith(prefix) and n.endswith(".pt"):
            tail = n[len(prefix):-len(".pt")]
            if tail.isdigit():
                steps.append(int(tail))
    return sorted(set(steps))


def fetch_resume_ckpt(env, out_dir, step, accelerator):
    """Fetch the best resume checkpoint from the Hub.

    Exact REPAIR/checkpoint-<step>.pt is preferred. A MAX_HOURS stop lands
    between checkpoint intervals, so state.json can name a step that was never
    saved; fall back to the newest checkpoint at or before that step (newest
    overall if every checkpoint is newer). Rank 0 downloads and writes its pick
    to a sidecar file so every rank resolves the same step. Returns the local
    path, or None if nothing could be fetched.
    """
    pick_path = os.path.join(out_dir, "hf_resume", "resume_step.txt")
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(pick_path), exist_ok=True)
        if os.path.exists(pick_path):
            os.remove(pick_path)  # never let a stale pick masquerade as this one
        steps = remote_ckpt_steps(env)
        pick = None
        if step in steps:
            pick = step
        elif steps:
            earlier = [s for s in steps if s < step]
            pick = earlier[-1] if earlier else steps[-1]
            print(f"[hf] checkpoint-{step}.pt is not on the Hub; resuming from "
                  f"the newest saved checkpoint, checkpoint-{pick}.pt", flush=True)
        if pick is None:
            print(f"[hf] no checkpoint found for step {step}", flush=True)
        else:
            local = os.path.join(out_dir, env["hf_dir"], f"checkpoint-{pick}.pt")
            try:
                if os.path.exists(local):
                    print(f"[hf] checkpoint-{pick}.pt already local; "
                          f"skipping download", flush=True)
                else:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo_id=env["hf_repo"],
                                    filename=f"{env['hf_dir']}/checkpoint-{pick}.pt",
                                    token=env["hf_token"],
                                    local_dir=os.path.join(out_dir, "hf_resume"))
                    print(f"[hf] downloaded checkpoint-{pick}.pt", flush=True)
                with open(pick_path, "w") as f:
                    f.write(str(pick))
            except Exception as e:
                print(f"[hf] checkpoint-{pick}.pt fetch failed: {e}", flush=True)
    accelerator.wait_for_everyone()
    if not os.path.exists(pick_path):
        return None
    try:
        with open(pick_path) as f:
            pick = int(f.read().strip())
    except ValueError:
        return None
    for base in (os.path.join(out_dir, "hf_resume"), out_dir):
        path = os.path.join(base, env["hf_dir"], f"checkpoint-{pick}.pt")
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------- export

HF_KEY_MAP = {
    "tok_emb.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
}
HF_SUB = {
    "attn_norm": "input_layernorm",
    "mlp_norm": "post_attention_layernorm",
    "attn": "self_attn",
    "mlp": "mlp",
}


def to_hf_state_dict(sd):
    """Our LLaMA state dict -> LlamaForCausalLM names (lm_head tied, dropped)."""
    out = {}
    for k, v in sd.items():
        if k in HF_KEY_MAP:
            out[HF_KEY_MAP[k]] = v
            continue
        if k == "lm_head.weight":
            continue  # tied to embed_tokens; the config tells HF to tie
        if k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            sub, leaf = rest.split(".", 1)
            if sub not in HF_SUB:
                raise KeyError(f"unmapped model key {k}")
            out[f"model.layers.{i}.{HF_SUB[sub]}.{leaf}"] = v
            continue
        raise KeyError(f"unmapped model key {k}")
    return out


def hf_config(cfg, theta, max_pos):
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": cfg["vocab_size"],
        "hidden_size": cfg["dim"],
        "intermediate_size": cfg["ffn_dim"],
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["n_heads"],
        "num_key_value_heads": cfg["n_kv_heads"],
        "head_dim": cfg["head_dim"],
        "hidden_act": "silu",
        "max_position_embeddings": max_pos,
        "initializer_range": 0.02,
        "rms_norm_eps": cfg["rms_eps"],
        "use_cache": True,
        "rope_theta": theta,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }


CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def export_hf(cfg, state_dict, theta, max_pos, out_dir, tokenizer_src):
    """Write the HF-format artifact set (safetensors + config + tokenizer)."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    save_file(to_hf_state_dict(state_dict), os.path.join(out_dir, "model.safetensors"),
              metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(hf_config(cfg, theta, max_pos), f, indent=1)
    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
                   "do_sample": True, "temperature": 0.8, "top_p": 0.95}, f, indent=1)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": max_pos,
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "chat_template": CHAT_TEMPLATE,
            "clean_up_tokenization_spaces": False,
        }, f, indent=1)
    if tokenizer_src and os.path.exists(tokenizer_src):
        shutil.copyfile(tokenizer_src, os.path.join(out_dir, "tokenizer.json"))


def verify_hf_export(export_dir, model, device, n=64):
    """Load the export with transformers and compare logits. Warn-only."""
    try:
        from transformers import LlamaForCausalLM
    except Exception as e:
        print(f"[export] transformers unavailable ({e}); skipping load check", flush=True)
        return None
    try:
        hf = LlamaForCausalLM.from_pretrained(export_dir, torch_dtype=torch.float32)
        hf.eval().to(device)
        ids = torch.randint(0, model.cfg["vocab_size"], (1, min(n, 64)), device=device)
        with torch.no_grad():
            ours = model(ids).float()
            theirs = hf(ids).logits.float()
        d = (ours - theirs).abs()
        print(f"[export] HF load check: max|dlogit| {d.max().item():.4f} "
              f"mean {d.mean().item():.5f}", flush=True)
        del hf
        return float(d.max().item())
    except Exception as e:
        print(f"[export] HF load check failed: {type(e).__name__}: {e}", flush=True)
        return None


def write_readme(path, arch, phases, step, tokens_seen, val_hist, final):
    with open(path, "w") as f:
        f.write(
            "# TinyBalls-V1 — tool-call repair (repair-tok-v1)\n\n"
            f"113M-param LLaMA (`{arch['dim']}`d, {arch['n_layers']}L, "
            f"{arch['n_heads']}/{arch['n_kv_heads']}-head GQA, vocab "
            f"{arch['vocab_size']}), repaired from the SFT checkpoint "
            f"`igidn/tinyballs-v1` `SFT/final.pt` on the targeted tool-use mix "
            f"`repair-tok-v1` (schema binding, abstention, error recovery, "
            f"grounded answers).\n\n"
            f"- phases: " + ", ".join(p["name"] for p in phases) + "\n"
            f"- steps: {step}, tokens: {tokens_seen/1e6:.1f}M\n"
            f"- val loss: {json.dumps(val_hist)}\n"
            f"- final: {json.dumps(final)}\n\n"
            "## Chat format\n\n"
            "```\n"
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n[{reasoning}] {content}<|im_end|>\n"
            "```\n\n"
            "Generation stops at `<|im_end|>` (id 2); eos (id 0) is the packing "
            "separator. Reasoning is plain text wrapped in `<think>...</think>`.\n\n"
            "## Files\n\n"
            "- `model.safetensors` / `config.json`: `transformers` "
            "`LlamaForCausalLM` (tied embeddings). Load with "
            "`AutoModelForCausalLM.from_pretrained(<this dir>)`; `tokenizer.json` "
            "is the exact tokenizer the model was trained with.\n"
            "- `final.pt`: same weights in this repo's own format "
            "(`llama.model.LLaMA`, state dict under `model`).\n"
            "- `checkpoint-<step>.pt`: resumable trainer state (model + "
            "optimizer + scaler), newest two kept.\n\n"
            "Caveat: training clamps SwiGLU pre-activations at 10.0; HF's "
            "`LlamaMLP` does not, so extreme activations can differ slightly.\n"
        )


# ---------------------------------------------------------------- main

def main():
    t_start = time.time()
    use_cuda = torch.cuda.is_available()
    accelerator = Accelerator(
        mixed_precision="fp16" if use_cuda else None,
        # accumulation is manual: accum changes per phase while Accelerator's
        # counter is fixed at construction
        gradient_accumulation_steps=1,
    )
    world = accelerator.num_processes
    rank = accelerator.process_index
    main_proc = accelerator.is_main_process
    torch.manual_seed(env_int("SEED", 1234))

    data_dir = os.environ["REPAIR_DATA_DIR"]
    tok_dir = os.environ.get("TOK_DATA_DIR", "")
    out_dir = os.environ.get("OUT_DIR", "/kaggle/working")
    run_out = os.path.join(out_dir, os.environ.get("HF_CKPT_DIR", "REPAIR"))
    os.makedirs(run_out, exist_ok=True)
    tokens_per_step = env_int("TOKENS_PER_STEP", 65536)
    micro_bs = env_int("MICRO_BS", 1)
    ce_chunk = env_int("CE_CHUNK", 2048)
    grad_ckpt_env = os.environ.get("GRAD_CKPT", "auto")
    ckpt_every = env_int("CKPT_EVERY", 250)
    eval_every = env_int("EVAL_EVERY", 250)
    max_hours = env_float("MAX_HOURS", 11.0)
    max_steps = env_int("MAX_STEPS", 0)
    seed = env_int("SEED", 1234)
    env = dict(
        hf_token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN", ""),
        hf_repo=os.environ.get("HF_CKPT_REPO", "igidn/tinyballs-v1"),
        hf_dir=os.environ.get("HF_CKPT_DIR", "REPAIR"),
        base_repo=os.environ.get("BASE_REPO", "igidn/tinyballs-v1"),
        base_ckpt_file=os.environ.get("BASE_CKPT_FILE", "SFT/final.pt"),
        base_ckpt_path=os.environ.get("BASE_CKPT_PATH", ""),
    )
    arch = dict(
        vocab_size=env_int("VOCAB", 49154), dim=env_int("D_MODEL", 768),
        n_layers=env_int("N_LAYERS", 12), n_heads=env_int("N_HEADS", 12),
        n_kv_heads=env_int("N_KV_HEADS", 4), head_dim=env_int("HEAD_DIM", 64),
        ffn_dim=env_int("FFN_DIM", 2048), rms_eps=env_float("RMS_EPS", 1e-5),
    )

    wandb = None
    if main_proc and os.environ.get("WANDB_API_KEY"):
        import wandb as _w
        _w.login(key=os.environ["WANDB_API_KEY"])
        wandb = _w.init(project=os.environ.get("WANDB_PROJECT", "tinyballs"),
                        name=os.environ.get("WANDB_NAME", "tinyballs-v1-repair"),
                        config=dict(arch=arch, world=world, micro_bs=micro_bs,
                                    tokens_per_step=tokens_per_step,
                                    phases=phase_specs(),
                                    base=f"{env['base_repo']}/{env['base_ckpt_file']}"))
    elif main_proc:
        print("[wandb] no WANDB_API_KEY; stdout logging only", flush=True)

    # ---- data: manifest, replay pool, per-phase seeded plans (every rank
    #      builds identical plans, which is what makes resume exact)
    with open(os.path.join(data_dir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("format", {}).get("vocab"):
        assert manifest["format"]["vocab"] == arch["vocab_size"], "vocab mismatch"
    replay_paths = []
    if any(s["replay"] > 0 for s in phase_specs()):
        replay_paths = sorted(glob.glob(os.path.join(tok_dir, "train", "*-p2-*.bin"))) \
            if tok_dir else []
        if not replay_paths:
            print("[data] no tok-mix p2 shards (TOK_DATA_DIR); replay disabled",
                  flush=True)

    phases = []
    for spec in phase_specs():
        accum = max(1, tokens_per_step // (micro_bs * spec["seq"] * world))
        slot = micro_bs * accum * world
        rep = ChunkPool(replay_paths, spec["seq"] + 1) if replay_paths else None
        t0 = time.time()
        pd = SFTPhaseData(data_dir, spec["buckets"], spec["seq"], replay=rep)
        segments = []
        if spec["cpt_tokens"] > 0:
            rows = pd.build_plan(seed=seed + 7, replay_frac=0.0,
                                 max_tokens=spec["cpt_tokens"])
            segments.append(dict(name="cpt", rows=rows, cpt=True))
        for e in range(spec["epochs"]):
            rows = pd.build_plan(seed=seed + e, replay_frac=spec["replay"])
            segments.append(dict(name=f"ep{e+1}", rows=rows, cpt=False))
        for s in segments:
            s["steps"] = len(s["rows"]) // slot
        spec.update(data=pd, segments=segments, accum=accum, slot=slot,
                    total_steps=sum(s["steps"] for s in segments),
                    grad_ckpt=(spec["seq"] > 8192 if grad_ckpt_env == "auto"
                               else grad_ckpt_env == "1"))
        if pd.n_oversize:
            print(f"[data] WARNING: {pd.n_oversize} {spec['name']} docs exceed "
                  f"seq {spec['seq']} and are dropped — is SEQ_{spec['name'].upper()} "
                  f"smaller than the bucket?", flush=True)
        if main_proc:
            print(f"[data] {spec['name']}: {pd.n_docs} docs / {pd.tokens/1e6:.1f}M tok | "
                  f"seq {spec['seq']} theta {spec['theta']:.0f} accum {accum} | "
                  + ", ".join(f"{s['name']}:{s['steps']}st({len(s['rows'])}r)"
                              for s in segments)
                  + f" | total {spec['total_steps']} steps ({time.time()-t0:.1f}s)",
                  flush=True)
        phases.append(spec)
    total_steps = sum(p["total_steps"] for p in phases)
    if max_steps:
        total_steps = min(total_steps, max_steps)
    if total_steps <= 0:
        raise SystemExit("no training steps — check PHASES/data")
    if main_proc:
        print(f"[plan] {total_steps} optimizer steps, "
              f"{tokens_per_step/1024:.0f}k tok/step, world {world}, "
              f"micro {micro_bs}, ce_chunk {ce_chunk}", flush=True)

    def locate(g):
        for p in phases:
            if g < p["total_steps"]:
                return p, g
            g -= p["total_steps"]
        return None, 0

    def seg_of(ph, ls):
        for s in ph["segments"]:
            if ls < s["steps"]:
                return s, ls
            ls -= s["steps"]
        return ph["segments"][-1], max(0, ls)

    # ---- model + fresh optimizer
    model = LLaMA(**arch, rope_theta=phases[0]["theta"],
                  grad_ckpt=phases[0]["grad_ckpt"])
    opt = torch.optim.AdamW(param_groups(model, wd=0.1), lr=phases[0]["lr"],
                            betas=(0.9, 0.95), fused=use_cuda)
    if main_proc:
        print(f"[model] {model.num_params()/1e6:.1f}M params, vocab {arch['vocab_size']}",
              flush=True)

    base_path = env["base_ckpt_path"]
    if base_path and os.path.exists(base_path):
        if main_proc:
            print(f"[base] local {base_path}", flush=True)
    else:
        if main_proc:
            from huggingface_hub import hf_hub_download
            dest = os.path.join(out_dir, "base")
            os.makedirs(dest, exist_ok=True)
            hf_hub_download(repo_id=env["base_repo"], filename=env["base_ckpt_file"],
                            token=env["hf_token"] or None, local_dir=dest)
            print(f"[base] downloaded {env['base_repo']}/{env['base_ckpt_file']}",
                  flush=True)
        accelerator.wait_for_everyone()
        base_path = os.path.join(out_dir, "base", env["base_ckpt_file"])
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    model.load_state_dict(base["model"], strict=True)
    del base
    accelerator.wait_for_everyone()
    if main_proc:
        print("[base] weights loaded (strict)", flush=True)
        shutil.rmtree(os.path.join(out_dir, "base"), ignore_errors=True)

    # ---- resume: local checkpoint first, else HF state.json -> checkpoint-step
    # rank 0 downloads; the file lands in a shared dir every rank can read
    if main_proc:
        fetch_remote_state(env, out_dir)
    accelerator.wait_for_everyone()
    state_path = os.path.join(out_dir, "hf_resume", env["hf_dir"], "state.json")
    state = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    if state and state.get("done"):
        if main_proc:
            print(f"[resume] repair SFT already complete at step {state.get('step')} "
                  f"({state.get('val_loss')}); nothing to do", flush=True)
        return
    start_step = int(state.get("step", 0)) if state else 0
    # local checkpoints are the freshest (written before their upload); pick
    # the newest local one that is at least as new as the remote state
    local_files = sorted(glob.glob(os.path.join(run_out, "checkpoint-*.pt")),
                         key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
    local_step = int(local_files[-1].rsplit("-", 1)[1].split(".")[0]) if local_files else -1
    ck_path = local_files[-1] if local_step >= start_step and local_files else None
    if ck_path is None and start_step > 0 and env["hf_repo"] and env["hf_token"]:
        ck_path = fetch_resume_ckpt(env, out_dir, start_step, accelerator)
    if ck_path is None and local_files:
        # Hub unreachable (or nothing uploaded yet): a local checkpoint behind
        # state.json still beats a restart from scratch
        ck_path = local_files[-1]
        print(f"[resume] Hub fetch unsuccessful; falling back to local "
              f"{os.path.basename(ck_path)} (state step {start_step})", flush=True)
    if ck_path is None and start_step > 0:
        raise SystemExit(f"state.json says step {start_step} but no checkpoint "
                         f"could be fetched; refusing to restart from scratch")
    tokens_seen, val_hist = 0, {}
    if ck_path and os.path.exists(ck_path):
        ck = load_ckpt(ck_path, model, opt, accelerator.scaler, "cpu")
        start_step = int(ck.get("step", start_step))
        tokens_seen = int(ck.get("tokens_seen", 0))
        val_hist = ck.get("val_hist", {})
        del ck
        print(f"[resume] {ck_path} -> step {start_step}/{total_steps}", flush=True)
    else:
        print("[resume] fresh repair run", flush=True)
    # newest checkpoint behind the resume point; state.json must never be
    # published past it (a MAX_HOURS stop lands between checkpoint intervals)
    last_state = dict(step=start_step, phase_name="", lr=0.0, tokens=tokens_seen,
                      val=None)
    start_step = min(start_step, total_steps)
    accelerator.wait_for_everyone()
    if main_proc:
        shutil.rmtree(os.path.join(out_dir, "hf_resume"), ignore_errors=True)

    model, opt = accelerator.prepare(model, opt)

    uploader = None
    if main_proc and env["hf_repo"] and env["hf_token"]:
        uploader = HFUploader(env["hf_repo"], env["hf_token"], env["hf_dir"])
        uploader.start()

    def publish_state(step, phase_name, lr, tokens, val, done=False):
        if uploader:
            uploader.submit(("state", dict(
                step=step, total_steps=total_steps, phase=phase_name, lr=lr,
                tokens_seen=tokens, val_loss=val, done=done,
                updated=time.strftime("%Y-%m-%d %H:%M:%S"))))

    def checkpoint(step, phase_name, lr, tokens, val):
        nonlocal last_state
        path = os.path.join(run_out, f"checkpoint-{step}.pt")
        save_ckpt(path, accelerator, model, opt,
                  dict(step=step, phase=phase_name, tokens_seen=tokens,
                       val_hist=val_hist, lr=lr))
        last_state = dict(step=step, phase_name=phase_name, lr=lr, tokens=tokens,
                          val=val)
        if uploader:
            uploader.submit(("ckpt", path, step))
            uploader.submit(("clean", step))
        publish_state(step, phase_name, lr, tokens, val)
        if main_proc:
            # local numbered copies pile up; keep the last two like HF does,
            # but never delete one the uploader has not confirmed yet
            locals_ = sorted(glob.glob(os.path.join(run_out, "checkpoint-*.pt")),
                             key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
            done = uploader.last_step if uploader else step
            for p in locals_[:-2]:
                if int(p.rsplit("-", 1)[1].split(".")[0]) <= done:
                    os.remove(p)

    ph_start, _ = locate(start_step)
    last_state["phase_name"] = ph_start["name"] if ph_start else ""
    last_state["val"] = val_hist.get(last_state["phase_name"])
    publish_state(start_step, ph_start["name"] if ph_start else "", 0.0, tokens_seen,
                  val_hist.get(ph_start["name"]) if ph_start else None)
    accelerator.wait_for_everyone()

    # ---- eval: masked val CE, weighted by train tokens, gathered over ranks
    def evaluate(ph):
        unw = accelerator.unwrap_model(model)
        unw.eval()
        dev = next(model.parameters()).device
        plan = ph["data"].build_val_plan()
        result = {}
        if plan:
            with torch.no_grad():
                s = torch.zeros((), dtype=torch.float32, device=dev)
                n = torch.zeros((), dtype=torch.float32, device=dev)
                for row in plan[rank::world]:
                    toks, mask = ph["data"].materialize(row)
                    x = torch.from_numpy(toks[:-1].astype(np.int64))[None].to(dev)
                    y = torch.from_numpy(toks[1:].astype(np.int64))[None].to(dev)
                    m = torch.from_numpy(mask[1:])[None].to(dev)
                    loss = model(x, y, loss_mask=m, ce_chunk=ce_chunk)
                    nt = float(m.sum().item())
                    s += loss.float() * nt
                    n += nt
            gathered = accelerator.gather(torch.stack([s, n])).reshape(world, 2)
            s, n = gathered[:, 0].sum(), gathered[:, 1].sum()
            result = {"val_loss": (s / n.clamp(min=1.0)).item(), "tokens": int(n.item())}
        unw.train()
        return result

    # ---- train
    step = start_step
    cur_phase = None
    tokens_win, t_win = 0, time.time()
    skip_streak = 0
    stop_reason = "completed"
    unw = accelerator.unwrap_model(model)

    def enter_phase(ph):
        nonlocal cur_phase
        if cur_phase is ph:
            return
        cur_phase = ph
        unw.cfg["rope_theta"] = ph["theta"]
        unw.grad_ckpt = ph["grad_ckpt"]
        unw._rope_key = None
        if main_proc:
            print(f"=== phase {ph['name']}: seq {ph['seq']} theta {ph['theta']:.0f} "
                  f"lr {ph['lr']:.1e} buckets {','.join(ph['buckets'])} ===", flush=True)

    if step > 0:
        ph0, _ = locate(step)
        enter_phase(ph0)

    try:
        while step < total_steps:
            ph, ls = locate(step)
            if ph is None:
                break
            enter_phase(ph)
            accum, slot = ph["accum"], ph["slot"]
            seg, seg_ls = seg_of(ph, ls)
            lr = lr_at(ls, ph["total_steps"], ph["lr"], ph["warmup"], ph["floor"])
            for g in opt.param_groups:
                g["lr"] = lr

            rows = [seg["rows"][seg_ls * slot + rank * (micro_bs * accum) + j]
                    for j in range(micro_bs * accum)]
            dev = next(model.parameters()).device
            loss_t = torch.zeros((), device=dev)
            skip_t = torch.zeros((), dtype=torch.int32, device=dev)

            opt.zero_grad(set_to_none=True)
            for j in range(accum):
                xb, yb, mb = [], [], []
                for r in rows[j * micro_bs:(j + 1) * micro_bs]:
                    toks, mask = ph["data"].materialize(r, cpt=seg["cpt"])
                    xb.append(toks[:-1])
                    yb.append(toks[1:])
                    mb.append(mask[1:])
                x = torch.from_numpy(np.stack(xb).astype(np.int64)).to(dev)
                y = torch.from_numpy(np.stack(yb).astype(np.int64)).to(dev)
                m = torch.from_numpy(np.stack(mb)).to(dev)
                ctx = (nullcontext() if j == accum - 1 or accum == 1
                       else accelerator.no_sync(model))
                with ctx:
                    loss = model(x, y, loss_mask=m, ce_chunk=ce_chunk)
                    if not torch.isfinite(loss):
                        skip_t += 1
                    # NaN*0 is still NaN, so rewrite the loss; inf activations
                    # still produce inf grads and the scaler backs off
                    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
                    accelerator.backward(loss / accum)
                loss_t += loss.detach()

            if any(p.grad is not None for p in model.parameters()):
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            step += 1
            tokens_seen += tokens_per_step
            tokens_win += tokens_per_step
            n_bad = int(skip_t.item())
            skip_streak = skip_streak + 1 if n_bad == accum else 0
            if skip_streak >= 20:
                raise SystemExit(f"loss diverged: 20 consecutive fully-skipped "
                                 f"steps ending at {step}")
            step_loss = accelerator.gather(loss_t).mean().item() / accum

            if step % 50 == 0 or step <= start_step + 5 or step == total_steps or n_bad:
                if main_proc:
                    tps = tokens_win / max(1e-9, time.time() - t_win)
                    msg = (f"[{step}/{total_steps}] {ph['name']} {seg['name']} "
                           f"loss {step_loss:.4f} lr {lr:.2e} {tps/1e3:.1f}k tok/s")
                    if n_bad:
                        msg += f" ({n_bad} bad)"
                    print(msg, flush=True)
                    if wandb:
                        wandb.log(dict(step=step, loss=step_loss, lr=lr,
                                       tokens_per_sec=tps, phase=ph["name"]), step=step)
                    t_win, tokens_win = time.time(), 0

            if step % eval_every == 0 or step == total_steps:
                vals = evaluate(ph)
                if main_proc and vals:
                    print(f"    val[{ph['name']}] {vals['val_loss']:.4f} "
                          f"({vals['tokens']} tok)", flush=True)
                    val_hist[ph["name"]] = vals["val_loss"]
                    if wandb:
                        wandb.log({f"val_{ph['name']}": vals["val_loss"]}, step=step)

            if step % ckpt_every == 0 or step == total_steps:
                checkpoint(step, ph["name"], lr, tokens_seen, val_hist.get(ph["name"]))

            if max_steps and step - start_step >= max_steps:
                stop_reason = f"MAX_STEPS {max_steps}"
                break
            if (time.time() - t_start) / 3600 > max_hours:
                stop_reason = f"MAX_HOURS {max_hours}"
                break
    except BaseException as e:
        import traceback
        traceback.print_exc()
        stop_reason = "exception"
        try:
            checkpoint(step, cur_phase["name"] if cur_phase else "",
                       0.0, tokens_seen,
                       val_hist.get(cur_phase["name"]) if cur_phase else None)
        except Exception as e2:
            print(f"[ckpt] emergency save failed: {e2}", flush=True)
        if uploader:
            uploader.drain(timeout=900)
        raise

    # ---- finalize: full artifacts + HF-format export
    # done=true only for a full-schedule run: a PHASES/MAX_STEPS-limited run
    # must stay resumable, otherwise it would block the real run later
    full_run = (max_steps == 0
                and {p["name"] for p in phases} == {"repair"})
    hf_check = None
    if stop_reason == "completed":
        ph = phases[-1]
        final = evaluate(ph)
        if main_proc:
            fv = final.get("val_loss")
            print(f"[final] {ph['name']} val {fv:.4f}" if fv is not None
                  else f"[final] {ph['name']} (no val rows)", flush=True)
            sd = move_state_dict(accelerator.get_state_dict(model))
            path = os.path.join(run_out, "final.pt")
            # repair fine-tuning runs at 8k but keeps the SFT model's 32768
            # context (same rope theta); do not downgrade the exported config
            max_pos = env_int("MAX_POS", 32768)
            torch.save(dict(model=sd, step=step, tokens_seen=tokens_seen,
                            val_hist=val_hist, cfg=dict(arch, rope_theta=ph["theta"]),
                            specials=dict(eos_id=0, im_start_id=1, im_end_id=2,
                                          tool_call_id=49152, tool_result_id=49153),
                            final=final), path)
            export_dir = os.path.join(run_out, "hf_export")
            export_hf(arch, sd, ph["theta"], max_pos,
                      export_dir, os.path.join(data_dir, "tokenizer.json"))
            hf_check = verify_hf_export(export_dir, accelerator.unwrap_model(model),
                                        "cuda" if use_cuda else "cpu")
            write_readme(os.path.join(export_dir, "README.md"), arch, phases,
                         step, tokens_seen, val_hist, final)
            if uploader:
                uploader.submit(("file", path, "final.pt"))
                for f in sorted(os.listdir(export_dir)):
                    uploader.submit(("file", os.path.join(export_dir, f), f))
            with open(os.path.join(run_out, "final_summary.json"), "w") as f:
                json.dump(dict(step=step, tokens_seen=tokens_seen, val_hist=val_hist,
                               final=final, hf_logit_maxdiff=hf_check),
                          f, indent=1)
        accelerator.wait_for_everyone()
    if main_proc:
        print(f"[done] {stop_reason} at step {step}/{total_steps}", flush=True)
        # publish the last *checkpointed* step, not the step the wall stopped
        # at: state.json must keep naming a checkpoint that exists
        publish_state(**last_state, done=(stop_reason == "completed" and full_run))
        if uploader:
            uploader.drain()
        if wandb:
            wandb.log({"final_step": step}, step=step)
            wandb.finish()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
