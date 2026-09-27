#!/usr/bin/env python
"""Chat with a finished TinyBalls SFT model — the end-to-end "is it a proper
LM" check.

Loads either the HF export (a directory with config.json + model.safetensors,
i.e. SFT/hf_export or the SFT/ files on the Hub) or a final.pt/checkpoint.pt in
this repo's own format, then runs a ChatML conversation and prints what the
model says. Stops at <|im_end|> (id 2), the generation stop.

  python sft/chat.py --model SFT/hf_export
  python sft/chat.py --model SFT/final.pt --temp 0.0 --prompt "hi"

Sampling defaults mirror generation_config.json: temp 0.8, top-p 0.95;
--temp 0 means greedy. Reasoning appears inline as <think>...</think>.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

IM_END = 2
DEFAULT_SYSTEM = "You are TinyBalls, a helpful small assistant."


def render_prompt(messages, add_generation_prompt=True):
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return out


class HFChat:
    def __init__(self, path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path)
        self.model.eval().to(device)
        self.device = device

    def reply(self, prompt, temp, top_p, max_new):
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            if temp <= 0:
                out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                          eos_token_id=IM_END, pad_token_id=0)
            else:
                out = self.model.generate(ids, max_new_tokens=max_new, do_sample=True,
                                          temperature=temp, top_p=top_p,
                                          eos_token_id=IM_END, pad_token_id=0)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text.split("<|im_end|>")[0]


class LocalChat:
    """The repo's own format (llama.model.LLaMA), manual sampling loop."""

    def __init__(self, path, device, tokenizer=None):
        from llama.model import LLaMA
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or dict(vocab_size=49154, dim=768, n_layers=12, n_heads=12,
                                    n_kv_heads=4, head_dim=64, ffn_dim=2048,
                                    rope_theta=10000.0, rms_eps=1e-5)
        self.model = LLaMA(**{k: cfg[k] for k in
                              ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
                               "head_dim", "ffn_dim", "rope_theta", "rms_eps")})
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.eval().to(device)
        self.device = device
        here = os.path.dirname(os.path.abspath(path))
        for cand in (tokenizer, os.path.join(here, "tokenizer.json"),
                     os.path.join(os.path.dirname(here), "tokenizer.json")):
            if cand and os.path.exists(cand):
                tok_path = cand
                break
        else:
            raise FileNotFoundError(
                "no tokenizer.json next to the checkpoint; pass --tokenizer "
                "(the file ships with the sft-tok-v1 dataset and the HF repo)")
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(tok_path)

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def reply(self, prompt, temp, top_p, max_new):
        ids = torch.tensor([self.encode(prompt)], dtype=torch.long, device=self.device)
        for _ in range(max_new):
            with torch.no_grad():
                logits = self.model(ids)[0, -1].float()
            if temp <= 0:
                nxt = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temp, dim=-1)
                k = int(top_p * probs.numel())
                if 0 < k < probs.numel():
                    cut = torch.topk(probs, k).values.min()
                    probs = probs.masked_fill(probs < cut, 0.0)
                nxt = int(torch.multinomial(probs / probs.sum(), 1))
            if nxt == IM_END:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]], device=self.device)], dim=1)
        return self.tok.decode(ids[0, len(self.encode(prompt)):].tolist(),
                               skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF export dir (config.json) or final.pt / checkpoint.pt")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json for the final.pt path (default: next to it)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--prompt", default=None, help="one-shot instead of REPL")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = args.model
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        bot = HFChat(path, device)
        print(f"[chat] HF export {path}")
    else:
        bot = LocalChat(path, device, args.tokenizer)
        print(f"[chat] local checkpoint {path}")

    messages = [{"role": "system", "content": args.system}]
    if args.prompt is not None:
        messages.append({"role": "user", "content": args.prompt})
        print(bot.reply(render_prompt(messages), args.temp, args.top_p, args.max_new))
        return
    print("type a message; empty line exits")
    while True:
        try:
            user = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            return
        messages.append({"role": "user", "content": user})
        reply = bot.reply(render_prompt(messages), args.temp, args.top_p, args.max_new)
        print(f"assistant> {reply}")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
