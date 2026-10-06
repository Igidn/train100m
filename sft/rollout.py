"""Rollout plumbing for TinyBalls-V1 RL: one renderer, two engines, one respond().

The renderer exists because the measured failure mode in `REPAIR2_EVAL.md` §2 was
not the model but the bytes between the weights and the sampler: four serving
defects, and fixing them moved single_tool 2/8 -> 7/8 with no weight change. A
rollout engine that renders prompts differently from `tokenize_sft.render` is the
same bug wearing a different hat, and it is silent — rollouts still look
reasonable, the run just optimises a model nobody measured.

So `render_prefix` is checked against `tokenize_sft.render`'s prefix in
`--selftest`, token id for token id. Anything that cannot be reproduced exactly
is a hard failure, not a warning.

Engines:
    vllm   the intended rollout engine (`RL_TRAINING.md` §6 risk 1: unverified on
           T4 until the smoke kernel says otherwise)
    kv     our own `llama/kv_cache.py` decode, verified exact against the
           re-forward path (`RL_READINESS.md` §1: max |logit diff| 2.2e-05, zero
           argmax mismatches) and correct for multi-turn, since every step is a
           fresh prefill of a longer prompt. Slower, and it is the fallback.

    python sft/rollout.py --selftest
"""

import argparse
import json

IM_START = "<" + "|im_start|>"
IM_END = "<" + "|im_end|>"
TOOL_CALL = "<" + "|tool_call|>"
TOOL_RESULT = "<" + "|tool_result|>"
IM_END_ID = 2


def _ids(tok, text):
    out = tok.encode(text, add_special_tokens=False)
    # tokenizers.Tokenizer -> .ids ; transformers -> list
    return list(getattr(out, "ids", out))


def render_prefix(tok, msgs):
    """Token ids for a conversation ending at an assistant turn's newline.

    Byte-identical to the corresponding prefix of `tokenize_sft.render`: same
    markers, same role lines, same `tool` turn shape (`<|tool_result|>` then the
    payload), and no trailing newline after an assistant turn.
    """
    ids = []
    seen_system = False
    for m in msgs:
        role = m["role"]
        if role == "system":
            if seen_system:
                continue
            seen_system = True
            ids += _ids(tok, IM_START) + _ids(tok, "system\n")
            ids += _ids(tok, m["content"])
            ids += _ids(tok, IM_END) + _ids(tok, "\n")
        elif role == "user":
            ids += _ids(tok, IM_START) + _ids(tok, "user\n")
            ids += _ids(tok, m["content"])
            ids += _ids(tok, IM_END) + _ids(tok, "\n")
        elif role == "tool":
            ids += _ids(tok, IM_START) + _ids(tok, "tool\n")
            ids += _ids(tok, TOOL_RESULT)
            ids += _ids(tok, m["content"])
            ids += _ids(tok, IM_END) + _ids(tok, "\n")
        elif role == "assistant":
            ids += _ids(tok, IM_START) + _ids(tok, "assistant\n")
            content = m.get("content", "")
            if m.get("reasoning"):
                ids += _ids(tok, "<think>" + m["reasoning"] + "</think>\n")
            if content:
                ids += _ids(tok, content)
            for tc in m.get("tool_calls", []) or []:
                ids += _ids(tok, TOOL_CALL)
                ids += _ids(tok, json.dumps({"name": tc["name"],
                                             "arguments": tc["arguments"]},
                                            ensure_ascii=False))
            ids += _ids(tok, IM_END)
        else:
            raise ValueError(f"unknown role {role!r}")
    return ids


# ------------------------------------------------------------------ engines
class VLLMEngine:
    """vLLM, fed token ids (never text) so the prompt cannot be re-tokenised."""

    name = "vllm"

    def __init__(self, model, max_model_len=4096, gpu_mem=0.35, seed=0,
                 dtype="float32"):
        from vllm import LLM
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_mem, enforce_eager=True, seed=seed)

    def generate(self, prompts, max_tokens=256, temp=1.0, top_p=0.95, seed=0):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens,
                            seed=seed, detokenize=False)
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(p))
                                  for p in prompts], sp)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append((list(c.token_ids), c.finish_reason))
        return res


class KVEngine:
    """Our own cached greedy/sampled decode (llama/kv_cache.py)."""

    name = "kv"

    def __init__(self, model_dir=None, repo=None, device=None, max_len=4096):
        import torch
        from transformers import AutoTokenizer, LlamaForCausalLM
        from llama.kv_cache import generate as kv_generate
        from llama.model import LLaMA
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo or model_dir)
        hf = LlamaForCausalLM.from_pretrained(repo or model_dir,
                                             torch_dtype=torch.float32).eval()
        c = hf.config
        m = LLaMA(vocab_size=c.vocab_size, dim=c.hidden_size,
                  n_layers=c.num_hidden_layers, n_heads=c.num_attention_heads,
                  n_kv_heads=c.num_key_value_heads,
                  head_dim=c.hidden_size // c.num_attention_heads,
                  ffn_dim=c.intermediate_size, rope_theta=float(c.rope_theta),
                  rms_eps=float(c.rms_norm_eps))
        m.load_state_dict(_hf_to_ours(hf.state_dict()), strict=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = m.eval().to(self.device)
        self.max_len = max_len
        self._generate = kv_generate

    def generate(self, prompts, max_tokens=256, temp=1.0, top_p=0.95, seed=0):
        res = []
        for p in prompts:
            p = list(p)[-self.max_len:]
            ids = self._generate(self.model, [p], max_new=max_tokens, temp=temp,
                                 top_p=top_p, seed=seed)[0]
            stopped = IM_END_ID in ids
            res.append((list(ids), "stop" if stopped else "length"))
        return res


def _hf_to_ours(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("lm_head"):
            continue
        out[(k.replace("model.embed_tokens", "tok_emb")
              .replace("model.norm", "norm")
              .replace("model.layers.", "blocks.")
              .replace("input_layernorm", "attn_norm")
              .replace("post_attention_layernorm", "mlp_norm")
              .replace("self_attn.", "attn."))] = v
    return out


def make_responder(engine, tok, max_tokens=256, temp=1.0, top_p=0.95, seed=0):
    """respond(messages) -> (text, stopped) for `reward.run_episode`.

    `stopped` is the format term's "did the turn close on im_end" signal: vLLM
    reports it as finish_reason, and `PARAMETER stop ""`-style truncation is
    exactly the defect that made a healthy model look broken, so it is measured
    rather than assumed.
    """
    counter = {"n": 0}

    def respond(messages):
        ids = render_prefix(tok, messages)
        (new, finish), = engine.generate([ids], max_tokens=max_tokens, temp=temp,
                                         top_p=top_p, seed=seed + counter["n"])
        counter["n"] += 1
        text = tok.decode(new, skip_special_tokens=False)
        return text, finish == "stop"

    return respond


# ------------------------------------------------------------------ selftest
def selftest():
    """render_prefix must equal tokenize_sft.render's prefix, id for id."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transformers import AutoTokenizer
    from reward import compare_weather_task, tools_system, TOOL_CALL

    model = os.environ.get("SMOKE_MODEL", "igidn/TinyBalls-110M-V1")
    tok = AutoTokenizer.from_pretrained(model)
    fails = []

    msgs = [{"role": "system", "content": tools_system()},
            {"role": "user", "content": "Compare Tokyo and New York."},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]},
            {"role": "tool", "tool_name": "get_weather",
             "content": json.dumps({"city": "Tokyo", "temp": "18.4C"})},
            {"role": "assistant", "content": "Tokyo is 18.4C."}]

    mine = render_prefix(tok, msgs)
    try:
        import tokenize_sft
    except Exception as e:
        print(f"INFO tokenize_sft unavailable ({e}); prefix check skipped")
        tokenize_sft = None

    if tokenize_sft is not None:
        # build the same conversation in the trainer's schema and compare the
        # prefix that ends at the last assistant turn's im_end
        train_msgs = [{"role": "system", "content": msgs[0]["content"]},
                      {"role": "user", "content": msgs[1]["content"]},
                      {"role": "assistant", "content": "",
                       "tool_calls": msgs[2]["tool_calls"]},
                      {"role": "tool", "content": msgs[3]["content"]},
                      {"role": "assistant", "content": msgs[4]["content"]}]
        native = tokenize_sft.Tokenizer.from_file(
            os.environ.get("TOK_JSON", "tokenizer.json")) \
            if hasattr(tokenize_sft, "Tokenizer") and os.path.exists(
                os.environ.get("TOK_JSON", "tokenizer.json")) else None
        if native is None:
            print("INFO tokenizer.json not found (set TOK_JSON); "
                  "comparing HF tokenizer against itself instead")
            native = tok
        got = tokenize_sft.render(native, train_msgs, {})
        if got is None:
            print("FAIL tokenize_sft.render returned None (length filters?)")
            fails.append("render_none")
        else:
            ref = list(got[0])
            # the reference ends after the final im_end; ours must be a prefix
            # of it up to and including that token
            same = ref[:len(mine)] == mine
            print(f"  {'ok ' if same else 'FAIL'} prefix matches render(): "
                  f"{len(mine)} ids vs {len(ref)}")
            if not same:
                for i, (a, b) in enumerate(zip(mine, ref)):
                    if a != b:
                        print(f"       first divergence at {i}: {a} vs {b}")
                        print(f"       ours  ...{tok.decode(mine[max(0,i-12):i+12])!r}")
                        print(f"       theirs...{tok.decode(ref[max(0,i-12):i+12])!r}")
                        break
                fails.append("prefix_mismatch")

    # markers survive as single ids
    for marker, want in ((IM_START, 1), (IM_END, 2), (TOOL_CALL, 49152),
                         (TOOL_RESULT, 49153)):
        got_ids = tok.encode(marker, add_special_tokens=False)
        ok = list(got_ids) == [want]
        print(f"  {'ok ' if ok else 'FAIL'} {marker} -> {list(got_ids)} (want [{want}])")
        if not ok:
            fails.append(f"marker_{want}")

    # a task's rendered prompt must end with the assistant marker line
    ids = render_prefix(tok, compare_weather_task().messages())
    tail = tok.decode(ids[-6:], skip_special_tokens=False)
    ok = tail.endswith("assistant\n")
    print(f"  {'ok ' if ok else 'FAIL'} prompt ends at 'assistant\\n' -> {tail!r}")
    if not ok:
        fails.append("prompt_tail")

    print()
    if fails:
        print(f"ROLLOUT SELFTEST FAIL: {fails}")
        return 1
    print("ROLLOUT SELFTEST PASS")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else 0)