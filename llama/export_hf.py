#!/usr/bin/env python
"""Export a TinyBalls-V1 trainer checkpoint to the HF artifact set.

`llama/train.py` only ever wrote resumable trainer state (`checkpoint-*.pt`,
`checkpoint-final.pt`: model + AdamW + RNG). The SFT and repair trainers grew
their own HF export (`export_hf` in `sft/train_sft.py` / `sft/train_repair.py`),
so the *base* checkpoint never got one. This is that export, for the base.

The key mapping and config are byte-identical to the ones the SFT trainers use,
so all three releases describe the same architecture the same way — the only
fields that legitimately differ are `rope_theta` and
`max_position_embeddings`, which must match the phase the weights were trained
at (base: 10k @ 2048; SFT replay: 500k @ 32768).

Usage:
  python llama/export_hf.py <checkpoint.pt> <out_dir> [--tokenizer tokenizer.json]
                            [--theta 10000] [--max-pos 2048] [--base]

`--base` writes the base-model flavor: eos/bos/pad = <|endoftext|> (id 0, the
document separator the pretraining actually emits) and no chat template. The
instruct flavor is eos = <|im_end|> (id 2) plus the ChatML template, matching
`SFT/` and `REPAIR2/`.

Then, if `transformers` and `llama.model` are importable, the export is loaded
back and compared against this repo's own forward pass. That check is the whole
point of doing the mapping here instead of trusting the shapes: a silently
mis-mapped tensor still loads, it just produces garbage.
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# --- mapping: our LLaMA state dict -> LlamaForCausalLM names.
# Kept identical to sft/train_sft.py and sft/train_repair.py. lm_head.weight is
# dropped because it is the same tensor as tok_emb.weight (tied); config.json
# carries tie_word_embeddings so HF re-ties it.
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
    out = {}
    for k, v in sd.items():
        if k in HF_KEY_MAP:
            out[HF_KEY_MAP[k]] = v
            continue
        if k == "lm_head.weight":
            continue
        if k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            sub, leaf = rest.split(".", 1)
            if sub not in HF_SUB:
                raise KeyError(f"unmapped model key {k}")
            out[f"model.layers.{i}.{HF_SUB[sub]}.{leaf}"] = v
            continue
        raise KeyError(f"unmapped model key {k}")
    return out


# Head counts are not recoverable from tensor shapes alone: q_proj is
# [n_heads*head_dim, dim] and only the product is visible (768 = 12*64 = 24*32).
# They are constants of the run-1 LLaMA spec — the only architecture that has
# ever been trained here — and llama/train.py hardcodes them when it builds the
# model, so assert the shapes match rather than guessing.
N_HEADS, N_KV_HEADS, HEAD_DIM = 12, 4, 64


def arch_from_state_dict(sd):
    """Recover the architecture from tensor shapes — no config to trust."""
    vocab, dim = sd["tok_emb.weight"].shape
    layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    q_out = sd["blocks.0.attn.q_proj.weight"].shape[0]
    kv_out = sd["blocks.0.attn.k_proj.weight"].shape[0]
    assert q_out == N_HEADS * HEAD_DIM, f"q_proj out {q_out} != {N_HEADS*HEAD_DIM}"
    assert kv_out == N_KV_HEADS * HEAD_DIM, f"k_proj out {kv_out} != {N_KV_HEADS*HEAD_DIM}"
    assert sd["blocks.0.attn.o_proj.weight"].shape == (dim, dim), "o_proj not square"
    return dict(vocab_size=vocab, dim=dim, n_layers=layers, n_heads=N_HEADS,
                n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
                ffn_dim=sd["blocks.0.mlp.gate_proj.weight"].shape[0])


def hf_config(arch, theta, max_pos):
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": arch["vocab_size"],
        "hidden_size": arch["dim"],
        "intermediate_size": arch["ffn_dim"],
        "num_hidden_layers": arch["n_layers"],
        "num_attention_heads": arch["n_heads"],
        "num_key_value_heads": arch["n_kv_heads"],
        "head_dim": arch["head_dim"],
        "hidden_act": "silu",
        "max_position_embeddings": max_pos,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-5,
        "use_cache": True,
        "rope_theta": theta,
        "rope_scaling": None,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "bos_token_id": 0,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }


CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def write_artifacts(out_dir, sd, arch, theta, max_pos, tokenizer_src, base):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    hf = to_hf_state_dict(sd)

    # tied head: assert it before dropping it, so a future run that unties
    # fails here instead of shipping a model with no output projection
    assert torch.equal(sd["lm_head.weight"], sd["tok_emb.weight"]), \
        "lm_head is not tied to tok_emb; refusing to drop it"

    save_file({k: v.contiguous() for k, v in hf.items()},
              os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    cfg = hf_config(arch, theta, max_pos)
    if not base:  # instruct flavor: generation stops at <|im_end|>
        cfg.update(bos_token_id=1, eos_token_id=2, pad_token_id=0)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    with open(os.path.join(out_dir, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": cfg["bos_token_id"],
                   "eos_token_id": cfg["eos_token_id"],
                   "pad_token_id": cfg["pad_token_id"],
                   "do_sample": True, "temperature": 0.8, "top_p": 0.95}, f, indent=1)

    tok = {"tokenizer_class": "PreTrainedTokenizerFast",
           "model_max_length": max_pos,
           "clean_up_tokenization_spaces": False}
    if base:
        # <|endoftext|> is what pretraining emits at document ends; there is no
        # turn structure, so no chat template and no im_end stop.
        tok.update(bos_token="<|endoftext|>", eos_token="<|endoftext|>",
                   pad_token="<|endoftext|>")
    else:
        tok.update(bos_token="<|im_start|>", eos_token="<|im_end|>",
                   pad_token="<|endoftext|>", chat_template=CHAT_TEMPLATE)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok, f, indent=1)

    if tokenizer_src:
        if not os.path.exists(tokenizer_src):
            raise SystemExit(f"tokenizer not found: {tokenizer_src}")
        shutil.copyfile(tokenizer_src, os.path.join(out_dir, "tokenizer.json"))
    return cfg


def verify(out_dir, sd, arch, theta, max_pos, tokenizer_json=None):
    """Two checks, and the first one is the one that matters.

    1. Mapping check (assert): with the SwiGLU clamp disabled, this repo's
       forward and `transformers`' LlamaForCausalLM must agree *exactly*. A
       mis-mapped tensor still loads and still has the right shapes — it just
       produces garbage — so this is the only check that actually proves the
       export is the same model.

    2. Caveat size (report): training clamps SwiGLU pre-activations at 10.0
       (`llama/model.py` SWIGLU_CLAMP); HF's LlamaMLP does not. So on real text
       the two disagree slightly, and the size of that disagreement belongs in
       the model card rather than in a silent assumption. It is not small:
       expect max|dlogit| ~1.5, argmax agreement ~98% on prose, worse on
       out-of-distribution token soup.
    """
    try:
        from transformers import LlamaForCausalLM
        import llama.model as M
        from llama.model import LLaMA
    except Exception as e:
        print(f"[verify] skipped ({type(e).__name__}: {e})")
        return None

    ours = LLaMA(rope_theta=theta, grad_ckpt=False, **arch)
    ours.load_state_dict(sd, strict=True)
    ours.eval()
    hf = LlamaForCausalLM.from_pretrained(out_dir, torch_dtype=torch.float32)
    hf.eval()

    def diff(ids):
        with torch.no_grad():
            a = ours(ids).float()
            b = hf(ids).logits.float()
        d = (a - b).abs()
        return (d.max().item(), d.mean().item(),
                (a.argmax(-1) == b.argmax(-1)).float().mean().item())

    def real_ids(n=64):
        if not tokenizer_json or not os.path.exists(tokenizer_json):
            return None
        from tokenizers import Tokenizer
        text = ("The capital of France is Paris. The capital of Germany is "
                "Berlin. Machine learning models are trained on large corpora "
                "of text to predict the next token in a sequence.\n\n"
                "def fibonacci(n):\n    if n <= 1:\n        return n\n"
                "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
                "import numpy as np\nx = np.array([1, 2, 3])\n")
        ids = Tokenizer.from_file(tokenizer_json).encode(text).ids[:n]
        return torch.tensor([ids or [0]])

    ok = True
    torch.manual_seed(0)
    probes = {"random ids": torch.randint(0, arch["vocab_size"], (1, 64))}
    r = real_ids()
    if r is not None:
        probes["real text"] = r

    was = M.SWIGLU_CLAMP
    try:
        M.SWIGLU_CLAMP = float("inf")  # match HF: no clamp
        for tag, ids in probes.items():
            mx, mean, agree = diff(ids)
            bad = mx > 1e-4
            ok &= not bad
            print(f"[verify] mapping  {tag:<11} max|dlogit| {mx:.2e} "
                  f"{'OK' if not bad else 'MISMATCH — mapping is wrong'}")
        M.SWIGLU_CLAMP = was
        if r is not None:
            mx, mean, agree = diff(r)
            print(f"[verify] caveat   real text   max|dlogit| {mx:.4f} "
                  f"mean {mean:.5f} argmax agree {agree:.3f}  "
                  f"(SwiGLU clamp {was} in training, absent in HF)")
    finally:
        M.SWIGLU_CLAMP = was
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--theta", type=float, default=10000.0)
    ap.add_argument("--max-pos", type=int, default=2048)
    ap.add_argument("--base", action="store_true")
    a = ap.parse_args()

    print(f"loading {a.checkpoint} ...", flush=True)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    for k in ("step", "phase", "tokens_seen"):
        if k in ck:
            print(f"  {k}: {ck[k]}")

    arch = arch_from_state_dict(sd)
    print(f"  arch: {arch}")
    params = sum(v.numel() for v in sd.values())
    uniq = params - sd["lm_head.weight"].numel()
    print(f"  params: {uniq/1e6:.1f}M unique ({params/1e6:.1f}M with tied head)")

    cfg = write_artifacts(a.out_dir, sd, arch, a.theta, a.max_pos, a.tokenizer, a.base)
    print(f"wrote {a.out_dir}: config theta={cfg['rope_theta']} "
          f"max_pos={cfg['max_position_embeddings']} eos={cfg['eos_token_id']}")
    tok_json = os.path.join(a.out_dir, "tokenizer.json")
    if not verify(a.out_dir, sd, arch, a.theta, a.max_pos, tok_json):
        raise SystemExit("EXPORT VERIFY FAILED — do not upload")


if __name__ == "__main__":
    main()
