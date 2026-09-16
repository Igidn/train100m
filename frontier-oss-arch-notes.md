# Frontier OSS architectures, September 2026

Notes from pulling three SOTA checkpoints off Hugging Face: DeepSeek-V4.1-Flash,
GLM-5.3-Flash, and Kimi-K3. Everything below comes from `config.json`, the
reference implementations shipped in the repos, and the model cards. No guesses
about weights we cannot see.

Sources are listed at the end.

---

## DeepSeek-V4.1-Flash

Released 2026-09-10. 552B backbone parameters, 8B active during prefill, 16B
during decode, 1M context, native text and image input. MoE with 384 routed
experts plus 1 shared, 6 routed per token.

### Layout

Causal encoder-decoder. 40 layers total, split as a 20-layer causal encoder
followed by a 20-layer decoder. The decoder does not build its own KV from its
own hidden states. It reads a global KV projected from the final encoder hidden
states. That is what keeps prefill cheap: 8B active instead of 16B.

### Text config

| Field | Value |
|---|---|
| hidden size | 5120 |
| layers | 40 |
| attention heads | 64 |
| KV | single latent head, head dim 512 |
| RoPE | 64 of 512 dims |
| q LoRA rank | 1280 |
| o LoRA rank | 1024, grouped over 8 groups |
| sliding window | 128 |
| compress ratios | 0,0 then 18 layers of 2, 20 layers of 1, 3 MTP layers of 0 |
| RoPE base | YaRN factor 16, original context 65536 |
| compressed KV theta | 160000 |
| RMS eps | 1e-20 |
| SwiGLU clamp | 10.0 |
| vocab | 129280 |

### Attention: CSA2

Every attention layer has two KV sources concatenated into one sparse attention
call:

1. A sliding window of 128 raw KV positions, stored in a ring buffer.
2. When the layer's compress ratio is above zero, a compressed KV of top-k
   selected latent positions. Index top-k is 512.

Layers are assigned static roles. A layer with a compress ratio of 2 pools 2
tokens into 1 latent, and only source layers actually compute and store that
latent. Other layers with the same ratio read the same cache. This is why
`kv_source_layer_ids` is just `[2, 8, 14, 20]`. Indexer keys are shared the same
way via `index_source_layer_ids`, and layers between sources reuse the top-k
index list rather than recomputing it.

The indexer runs in two levels. The first level, at layer 20, scores blocks of
8 positions and keeps 2048 blocks as a candidate pool. Later indexing layers
score only inside those blocks. The indexer itself is a small attention: 32
heads of dim 128, fp4-quantized q and k, ReLU scores weighted by a learned
per-head projection.

KV cache totals 890 bytes per token. Cache entries are fp4 E2M1 with one E4M3
scale per 16 channels. That is roughly 1/4 of V4-Flash and 1/437 of V1.

### Compressor

Pools `ratio` consecutive tokens into one latent. Both the latent and the
pooling gate are projected from the stream, and the group is combined with a
softmax over the group. The latent is normalized with RMSNorm. Rotating
positions for compressed latents use their own theta of 160000, because one
latent stands for `ratio` tokens, so its effective positions are further apart.
A latent is treated as occupying the position of the first token of its group.
During decode the partial group is carried in state buffers and only emitted
when the group fills.

### Engram

A lexical n-gram memory, 196B parameters total, sparsely accessed by lookup.
Present at layers 1 and 14.

The scheme:

- Token ids are first mapped to a compressed vocabulary of 99,092 entries. The
  normalizer is NFKC, NFD, strip accents, lowercase, collapse whitespace. So
  " The", "the" and "THE" all hash identically.
- Each position hashes n-grams ending there, sizes 2 through 4.
- Each layer has its own hash table. Each n-gram size and head pair owns a
  disjoint bucket range sized by a unique prime drawn from 16M upward, so the
  ranges never overlap.
- The hash is a rolling XOR of token id times an odd random multiplier, one
  multiplier per layer and lookback.
- The two tables hold 384,006,168 and 384,016,682 rows of 256 dimensions,
  stored fp8 with block scales.

A lookup produces `n_hash_cols` rows. A projection turns them into one key per
hyper-connection copy plus one shared value. The gate is a normalized dot
product between the residual stream and the key, put through
`sigmoid(signed_sqrt(abs(dot)))`. The value is added to the stream scaled by
that gate. If the stream and the n-gram key agree, the memory writes.

### mHC, manifold-constrained hyper-connections

The residual stream is not one vector. It is 4 parallel copies, `hc_mult: 4`.

Each sublayer, attention and FFN separately, computes three coefficient sets
from the stream itself:

- `pre`, a collapse of the 4 copies into one sublayer input, `sigmoid` plus eps.
- `post`, an expansion weight, `2 * sigmoid`.
- `comb`, a 4x4 mixing matrix between the copies and the residual.

All 24 coefficients `(2 + 4) * 4` come from one projection of the flattened
stream, normalized by the root mean square. The `comb` matrix starts as a row
softmax and is then normalized alternately by row and column for 20 Sinkhorn
iterations, which makes it approximately doubly stochastic. That constraint
keeps the residual stream from growing or collapsing across depth. Coefficients
computed by one sublayer are consumed by the next.

### MTP and DSpark

Three next-token prediction layers are appended after the backbone. They read
the attention inputs of layers 37, 38 and 39, not the outputs.

DSpark is the speculative decoding head. It drafts a block of 5 tokens starting
from a noise token, attends over the main model's sliding window KV plus the
draft prefix, and adds a rank-256 Markov embedding to the logits at each draft
step. A confidence head scores drafts for verification. The repo implements the
forward pass only; the verification loop is left out.

### Precision

Weights are fp8 E4M3 with 32x32 blocks and ue8m0 scales. Routed experts are fp4.
Activations quantize dynamically. This is serving precision, not something a
Turing card can run.

### Training

45T tokens. Sparse attention trained at 64K context, then extended to 1M during
the last 34T tokens. Post-training is SFT, RL, then on-policy distillation. The
instruct model exposes a reasoning effort integer from 1 to 100.

---

## GLM-5.3-Flash

320B total, 18B active. Native multimodal, 1M context. Trained on a 30T token
corpus. First GLM with a hybrid attention stack.

### Layout

45 layers. 34 are KDA linear attention, 11 are sparse full attention at layers
3, 7, 11, ..., 43, meaning every 4th layer. The first 3 layers have dense MLPs,
the remaining 42 are MoE.

### Text config

| Field | Value |
|---|---|
| hidden size | 4096 |
| attention heads | 64 |
| qk head dim | 256, all NoPE, rope dim 0 |
| v head dim | 256 |
| q LoRA rank | 1536 |
| KV LoRA rank | 512 |
| indexer | 32 heads x 128 dim, top-k 2048 |
| index key pool | 4, compressed, always keeps the tail |
| KDA | 64 heads x 128 dim, short conv 4, gate lower bound -5.0 |
| MoE | 288 routed, 8 per token, 1 shared, intermediate 2048 |
| router | sigmoid scoring, noaux_tc, routed scale 2.5 |
| mHC | enabled, hc_mult 4, 20 Sinkhorn iterations |
| SwiGLU clamp | 10.0 |
| RMS eps | 1e-5 |
| MTP layers | 1 |
| vocab | 154880 |

### Attention

The full attention layers use MLA with q LoRA 1536 and KV LoRA 512. The query
and key head dim is 256 and there is no RoPE at all, `qk_rope_head_dim: 0`. The
11 full layers are all of type `deepseek_sparse_attention`, so GLM adopted the
DeepSeek sparse indexer for them. The indexer's key pool is compressed 4-to-1
and always includes the tail block, and the indexer RoPE is interleaved.

The KDA layers are the Kimi Linear design. The forget gate is per-channel with a
lower bound of -5.0 on the gate input.

### mHC

Identical hyperparameters to DeepSeek V4.1: 4 copies, 20 Sinkhorn iterations,
eps 1e-6.

### Training and deployment

fp8 dynamic quantization, E4M3, 128x128 blocks. The model card cites the GLM-5
technical report, arXiv:2602.15763. Thinking budget is controlled with
`reasoning_effort` at low, high, or max.

---

## Kimi-K3

2.8T total, 104B active, 93 layers, 1M context, native text and image. The
largest open-weight model to date. Claims 2.5x scaling efficiency over K2.

### Layout

69 KDA layers and 24 gated MLA layers. Exactly one dense layer, the first. MoE
with 896 routed experts, 16 active per token, plus 2 shared experts. Latent MoE
dimension 3584, expert intermediate 3072, which is the "Stable LatentMoE" part
of the announcement. Everything else is the same recipe as the other two: 1M
context, gated MLA, hybrid attention.

### Text config

| Field | Value |
|---|---|
| hidden size | 7168 |
| attention heads | 96 |
| q LoRA rank | 1536 |
| KV LoRA rank | 512 |
| qk nope dim | 128 |
| qk rope dim | 64, shared across heads |
| v head dim | 128 |
| KDA | 96 heads x 128 dim, short conv 4 |
| activation | SiTU-GLU, betas 4.0 and 25.0 |
| attention residuals | block size 12, per the model card only |
| MoE | 896 routed, 16 per token, 2 shared, latent dim 3584 |
| router | sigmoid, noaux_tc |
| vocab | 163840 |

### KDA, Kimi Delta Attention

The HF docstring says it plainly: the same as a gated delta net, but the decay
is per-channel instead of per-token.

Per layer:

- q, k, v projections, concatenated and passed through one depthwise causal
  conv1d of kernel size 4 with the activation function applied. This is the
  only source of local position mixing in a KDA layer.
- Forget gate per channel: `g = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)`,
  with `A_log` per head and `dt_bias` per channel. The softplus is written with
  a guard for large inputs.
- Input gate for the delta rule: `beta = sigmoid(b_proj(x))`.
- The delta rule itself, with q and k L2-normalized inside the kernel. The repo
  shows both a chunked and a single-token recurrent implementation, with a
  Triton path from the `fla` library and a pure PyTorch fallback.
- Output goes through a gated RMSNorm. The gate is projected by a second
  two-step low-rank path, `g_b(g_a(x))`.

The config also carries `gate_lower_bound: -5.0` and `use_full_rank_gate: true`.
The HF forget gate module notes it is the same as GLM's with the lower bound
removed, so the released text code and the config do not fully agree here.

### Gated MLA

Standard MLA with q LoRA 1536 and KV LoRA 512. The rotated key, 64 dims, is
shared across all heads. The nope part is 128 dims. `mla_use_output_gate` is set
in the config. Note the caveat below: the open text implementation does not
reference that flag.

### Attention residuals

The model card lists AttnRes with block size 12 as a core architectural change.
The config has `attn_res_block_size: 12`. The open text implementation in
transformers uses plain pre-norm residuals and never references the field, so
there is no public reference code for AttnRes yet. Treat it as unverified.

Same caveat for `mla_use_nope`, which appears in the config but is not read by
the released text implementation.

### Precision

MXFP4 weights and MXFP8 activations, with quantization-aware training. Vision
tower is MoonViT-V2 at 401M parameters.

---

## What all three agree on

These are not independent choices. A 2026 frontier stack looks like this.

**Full attention is no longer the default.** Kimi runs 69:24 KDA to full, GLM
34:11, DeepSeek replaces most layers with sliding window plus compressed sparse.
A 3:1 or 4:1 ratio of cheap to full layers is the norm.

**Exact position encoding is on the way out.** GLM sets the RoPE dimension to
zero. DeepSeek keeps RoPE on 64 of 512 dims and gives compressed latents their
own theta. Kimi uses one shared 64-dim rotated key. Local order is carried by
short convolutions and recurrent state instead.

**KV cache size is the optimization target.** KV sharing across layers, latent
KV, fp4 cache, 890 bytes per token. These decisions are about serving long
context, not about training quality.

**Sparse selection is learned, not fixed.** Learned indexers pick top-k
positions, with a two-level candidate scheme in DeepSeek and a compressed key
pool in GLM.

**Multi-token prediction is architecture.** DeepSeek ships 3 MTP layers plus a
block drafter, GLM ships 1. The layers double as speculative decoding.

**The plain residual is gone.** DeepSeek and GLM run manifold-constrained
hyper-connections with 4 copies and Sinkhorn-normalized mixing. Kimi runs
something called attention residuals. All three changed the residual stream in
the same generation.

**Stability defaults hardened.** SwiGLU clamped at 10, learned attention sinks,
per-channel gate bounds, tiny or fp32-precision normalization, doubly
stochastic mixing matrices. These are cheap and they show up everywhere.

**DeepSeek's Engram is unique.** The other two have no lexical memory module.

---

## What transfers to 100M parameters on 2x T4

### Hardware filter

T4 is Turing, sm75.

- No bf16, no TF32. fp16 AMP with GradScaler.
- No fp8 or fp4 tensor cores. All quantization in these models is irrelevant,
  except as something to read.
- No FlashAttention-2. PyTorch SDPA's memory-efficient backend works on sm75.
- Triton runs, so `fla` KDA kernels should work, but expect low MFU. The pure
  PyTorch chunked fallback in the Kimi implementation is the safe path.
- 16GB per card, 12h sessions, 30 GPU hours per week, 20GB disk.

### Take

**KDA hybrid at 3:1.** The main architectural story of the generation, and one
of the few pieces with public, readable training code. Test it rather than
copying it: at 1k to 2k sequence length the asymptotic advantage does not
appear, so any win comes from the inductive bias, not from FLOPs.

**Short conv of kernel 4 on the linear layers and a gated RMSNorm on the
output.** Both are in the KDA code and cost almost nothing.

**NoPE in the full-attention layers, as an ablation against RoPE.**

**QK-norm, a learned attention sink, and an output gate on attention.** All
three are small changes with reference implementations in the pulled code.

**SwiGLU clamp at 10.** Free stability for fp16.

**A slimmed Engram.** Hashed 2-gram and 3-gram tables with the stream-key gate,
at two layers. Costs memory and lookup time, no matmul FLOPs, and targets
exactly what a 100M model is bad at: local statistics and rare token pairs.
DeepSeek's `engram.py` is 184 lines and portable.

**One MTP layer.** A second loss term for one layer of compute.

### Leave

- MoE, by constraint.
- fp8 and fp4. No hardware support.
- Sparse indexers, candidate blocks, top-k gathers. Pure PyTorch gather on T4
  will eat the session budget, and the payoff only exists at long context.
- MLA latent compression at dim 768. The KV saving is a serving concern. GQA
  plus QK-norm is stronger at this scale.
- mHC and AttnRes in phase one. Both are interesting and both are risky at 4x
  smaller width than the smallest model that shipped them. Second phase.
- Vision, 1M context, DSpark, causal encoder-decoder.

---

## Proposal: TinyBalls-V1

A 100M-class, non-MoE model that adopts the transferable half of the 2026
recipe, sized for two T4s.

| Component | Choice |
|---|---|
| vocab | 32768, embeddings tied to the head |
| width | 768 |
| layers | 12 |
| pattern | KDA at layers 1-3, 5-7, 9-11; full attention at 4, 8, 12 |
| KDA | 12 heads x 64 dim, short conv 4, per-channel forget gate, gated RMSNorm output |
| full attention | 12 q heads, 4 KV heads, head dim 64, QK-norm, attention sink, output gate, NoPE |
| FFN | SwiGLU, intermediate 2048, clamp 10 |
| Engram | layers 1 and 7, 2-gram and 3-gram, 4 heads x 64 dim, 2^17 rows per table |
| MTP | 1 extra layer, shared embeddings and head |
| params | about 110M active, about 17M more in engram tables |
| precision | fp16 AMP, fp32 norms, GradScaler |
| training | DDP across 2 T4s, sequence 1024, about 2B tokens |

### Validation plan

Three runs, identical data, token budget, and active parameter count within a
few percent:

1. LLaMA baseline: RoPE, GQA, SwiGLU, no extras.
2. Hybrid stack: KDA 3:1, NoPE full layers, gating, clamp, sink.
3. Full stack: run 2 plus Engram and MTP.

Metrics: validation loss on a held-out shard, a small downstream suite, and a
synthetic induction and copy probe, because linear attention is known to be
weakest at exact recall and that is where a hybrid would show damage first.

Honest expectation: at 2B tokens the deltas will be small and noisy. The Engram
is the most likely positive, MTP a small positive, KDA possibly neutral or
negative at short context. The value is the measurement, not the score.

### Runtime plan

Each run is roughly 11 hours at an estimated 25 to 35 TFLOPS combined, so one
run per session with hard checkpointing every 1k steps. Checkpoints push to
Hugging Face Hub during the run and resume from a Kaggle Dataset input. Two to
three runs fit a week of quota.

---

## Sources

- DeepSeek-V4.1-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
  - `config.json`
  - `inference/model.py` (attention, compressor, indexer, Engram, mHC, DSpark)
  - `inference/engram.py` (hash construction and lookup)
  - `inference/kernel.py` (`hc_split_sinkhorn`, fp4 and fp8 GEMMs)
  - `DeepSeek_V41_Tech_Report.pdf`
- GLM-5.3-Flash: https://huggingface.co/zai-org/GLM-5.3-Flash
  - `config.json`
  - GLM-5 technical report, arXiv:2602.15763
  - blog: https://z.ai/blog/glm-5.3-flash
- Kimi-K3: https://huggingface.co/moonshotai/Kimi-K3
  - `config.json`
  - `configuration_kimi_k3.py`
- Kimi Linear text implementation, from transformers:
  `src/transformers/models/kimi_linear/modeling_kimi_linear.py`
  - `KimiLinearAttention`, `KimiLinearForgetGate`,
    `recurrent_kimi_delta_attention`, `KimiLinearDeltaAttention`
- Kimi Linear paper: the KDA design originates there.

Local copies of the reference files were read from `/tmp/hf-look/` during the
analysis session.
