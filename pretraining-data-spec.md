# Pretraining data spec

Target: ~110M parameter, non-MoE, English-only model. 2B tokens total, split as
1.5B broad pretraining plus a 500M quality anneal. The goal at this size is
behavioral range, not benchmark scores. No news. No TinyStories. No
multilingual.

Status: mix approved. Sources and pipeline below are the plan of record.

---

## 1. Final mix

### Phase 1, broad pretraining, 1.5B tokens

| Domain          |   Tokens |
| --------------- | -------: |
| Web             | **670M** |
| Books           | **150M** |
| Wikipedia       | **130M** |
| Code            | **120M** |
| Math            | **110M** |
| Synthetic prose |  **80M** |
| Papers          |  **80M** |
| QA & forums     |  **80M** |
| Micro-domains   |  **80M** |
| **Total**       | **1.5B** |

### Phase 2, quality anneal, 500M tokens

| Domain                                 |   Tokens | Purpose                               |
| -------------------------------------- | -------: | ------------------------------------- |
| **Instructional / high-quality prose** | **100M** | Prepare behavioral distribution       |
| **Math quality**                       |  **90M** | Clean up mathematical representations |
| **Code quality**                       |  **80M** | Structured/procedural language        |
| **Math solutions**                     |  **70M** | Problem to procedure to answer        |
| **Wiki + QA refresh**                  |  **70M** | Reinforce factual knowledge           |
| **OER / OCR books**                    |  **40M** | Educational and long-form language    |
| **Reasoning traces**                   |  **25M** | Reasoning patterns                    |
| **Agentic**                            |  **25M** | Sequential/action-oriented behavior   |
| **TOTAL**                              | **500M** |                                       |

---

## 2. Source mapping

Every line item maps to concrete Hugging Face datasets and configs. Token
counts are targets measured with the final tokenizer, not estimates from file
size.

### Phase 1

**Web, 670M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceFW/fineweb-edu | sample-10BT | 670M |

**Books, 150M**

| Source | Config | Tokens |
|---|---|---|
| common-pile/project_gutenberg | default | 55M |
| common-pile/pre_1929_books | default | 40M |
| common-pile/doab | default | 30M |
| common-pile/library_of_congress | default | 25M |

**Wikipedia, 130M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceFW/finewiki | en | 130M |

**Code, 120M**

| Source | Config | Tokens |
|---|---|---|
| openbmb/UltraData-Code | UltraData-Code-L2 py | 70M |
| codeparrot/codeparrot-clean | default | 50M |

**Math, 110M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceTB/finemath | finemath-3plus | 70M |
| open-web-math/open-web-math | default | 20M |
| math-ai/AutoMathText | arxiv-0.80-to-1.00 | 20M |

**Synthetic prose, 80M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceTB/cosmopedia | web_samples_v2 | 25M |
| HuggingFaceTB/cosmopedia | stories | 20M |
| HuggingFaceTB/cosmopedia | openstax | 10M |
| HuggingFaceTB/cosmopedia | stanford | 10M |
| HuggingFaceTB/cosmopedia | wikihow | 10M |
| HuggingFaceTB/cosmopedia | khanacademy | 5M |

**Papers, 80M**

| Source | Config | Tokens |
|---|---|---|
| common-pile/peS2o | default | 40M |
| common-pile/arxiv_abstracts | default | 20M |
| EleutherAI/proof-pile-2 | arxiv | 20M |

**QA and forums, 80M**

| Source | Config | Tokens |
|---|---|---|
| common-pile/stackexchange | default | 50M |
| allenai/dolmino-mix-1124 | stackexchange | 15M |
| common-pile/ubuntu_irc | default | 15M |

**Micro-domains, 80M**

| Source | Config | Tokens |
|---|---|---|
| common-pile/uspto | default | 15M |
| common-pile/caselaw_access_project | default | 15M |
| common-pile/regulations | default | 10M |
| common-pile/usgpo | default | 10M |
| common-pile/youtube | default | 10M |
| common-pile/foodista | default | 10M |
| common-pile/python_enhancement_proposals | default | 5M |
| common-pile/biodiversity_heritage_library | default | 5M |

### Phase 2

**Instructional / high-quality prose, 100M**

| Source | Config | Tokens |
|---|---|---|
| allenai/dolmino-mix-1124 | flan | 40M |
| TIGER-Lab/WebInstructSub | default | 30M |
| HuggingFaceTB/cosmopedia | openstax, stanford, khanacademy | 30M |

**Math quality, 90M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceTB/finemath | finemath-4plus | 50M |
| math-ai/AutoMathText | web-0.80-to-1.00 | 25M |
| HuggingFaceTB/finemath | infiwebmath-4plus | 15M |

**Code quality, 80M**

| Source | Config | Tokens |
|---|---|---|
| OpenCoder-LLM/opc-annealing-corpus | algorithmic_corpus | 20M |
| OpenCoder-LLM/opc-annealing-corpus | synthetic_code_snippet | 20M |
| OpenCoder-LLM/opc-annealing-corpus | synthetic_qa | 10M |
| nvidia/OpenCodeReasoning | split_0, split_1 | 30M |

**Math solutions, 70M**

| Source | Config | Tokens |
|---|---|---|
| nvidia/OpenMathInstruct-2 | default | 40M |
| open-r1/OpenR1-Math-220k | all | 30M |

**Wiki and QA refresh, 70M**

| Source | Config | Tokens |
|---|---|---|
| HuggingFaceFW/finewiki | en | 40M |
| common-pile/stackexchange_filtered | default | 30M |

**OER and OCR books, 40M**

| Source | Config | Tokens |
|---|---|---|
| common-pile/libretexts_filtered | default | 10M |
| common-pile/oercommons_filtered | default | 10M |
| allenai/olmOCR-mix-0225 | train-s2pdf, train-iabooks | 15M |
| common-pile/pressbooks_filtered | default | 5M |

**Reasoning traces, 25M**

| Source | Config | Tokens |
|---|---|---|
| open-thoughts/OpenThoughts-114k | default | 25M |

**Agentic, 25M**

| Source | Config | Tokens |
|---|---|---|
| nebius/SWE-agent-trajectories | default | 8M |
| bigcode/commitpackft | default | 8M |
| glaiveai/glaive-function-calling-v2 | default | 5M |
| SWE-Gym/OpenHands-Sampled-Trajectories | default | 4M |

### Deliberately excluded

News. `vblagoje/cc_news` and friends are out per decision.

TinyStories. Out per decision. The synthetic prose budget goes to Cosmopedia
instead.

Multilingual sources. English only.

Gated code corpora. `bigcode/starcoderdata` and `bigcode/the-stack-v2` are
better code data than what we chose, but they need terms acceptance and add
friction. Swap-ins if we ever want them.

---

## 3. Tokenizer

Decision: SmolLM2's tokenizer, byte-level BPE, 49,152 vocab, from
`HuggingFaceTB/SmolLM2-1.7B`. Files: `tokenizer.json`, `vocab.json`,
`merges.txt`. Loads with `AutoTokenizer`, no SentencePiece, no
`trust_remote_code`.

Survey method: 762 text-generation models in the 1B to 4B range
(`filter=text-generation`, `num_parameters=min:1000000000,max:4000000000`),
sorted by downloads, likes, and recency, then tokenizer files pulled from every
base model worth considering. Every 2025-2026 release has moved to a 100k to
260k vocab: SmolLM3 128,256, Qwen3 151,936, Granite 4.1 100,352, MiniCPM5
130,560, LFM2.5 128,000, Falcon3 131,072, Nemotron-3 131,072, Ministral-3
131,072, OLMo-2 100,352, Phi-4-mini 200,064, Gemma 2 256,000. For an
English-only 110M model that is dead weight in the embedding.

The only maintained small-vocab options, measured on a stratified sample of the
actual mix:

| Tokenizer | Vocab | chars/token | Used by |
|---|---|---|---|
| SmolLM2 | 49,152 | 3.93 | SmolLM2-1.7B, same recipe as Granite 3.1-2B |
| Mistral v0.1 | 32,000 | 3.63 | Zamba2-1.2B |
| Llama-2 | 32,000 | 3.56 | TinyLlama-1.1B, OpenELM, Phi-3 |
| Qwen3 | 151,936 | 4.14 | Qwen3-1.7B |

At 2B tokens, that is 7.87GB of text with SmolLM2, 7.27GB with Mistral, 7.13GB
with Llama-2, 8.27GB with Qwen3. The 151k vocab buys 5% more text for triple
the embedding, which is why it loses. At width 768, the tied embedding is 37.7M
params for 49k against 24.6M for 32k, taking the model from about 110M to about
123M. The extra FLOPs land in the output projection, roughly 5% per token, for
10% more text. 49k is the right side of that trade.

Training our own tokenizer was considered and rejected. A bespoke English 40k
BPE would gain a few percent over SmolLM2 at best, and we would own the risk.
SmolLM2's tokenizer was built for exactly this regime: small English models on
FineWeb-Edu, Python-Edu, and Cosmopedia, which is most of our mix. Its
pre-tokenizer is Digits with individual digit splitting plus the byte-level
regex, so math and code tokenize cleanly.

Special tokens, all defined before pre-tokenization so ids stay stable forever:

```
<|endoftext|>  id 0, document separator during pretraining, EOS later
<|im_start|>   id 1, ChatML role start for phase 2 and SFT
<|im_end|>     id 2, ChatML role end
ids 3-16       repo, file, issue, and jupyter specials, kept, useful for code
               and agentic formatting in phase 2
```

Added now, taking the vocab to 49,154:

```
<|tool_call|>
<|tool_result|>
```

---

## 4. Pipeline

1. **Fetch.** Stream each source from Hugging Face at a pinned revision. Record
   the commit hash in the manifest. Extract plain text into a common record:
   `{text, source, domain, phase}`.

2. **Filter.** Drop documents under 50 tokens. Cap document length at 8192
   tokens, splitting longer documents at paragraph boundaries. Run language ID
   on sources that can contain non-English text, which includes the Common Pile
   slices and commit data. Drop non-English documents.

3. **Deduplicate.** FineWeb-Edu, FineWiki, and Cosmopedia are already deduped.
   For the rest, exact-match dedup plus MinHash near-dedup within each source.
   No cross-domain dedup.

4. **Sample.** Fetch a pool of 2-3x the target per source, tokenize with the
   final tokenizer, then sample documents uniformly at random until the token
   target is hit. Fixed seed per source. This makes token counts exact instead
   of approximate.

5. **Split.** Hold out 2M tokens stratified by domain for validation. Keep the
   split stable across phases.

6. **Decontaminate.** Remove any document sharing an 8-gram with HellaSwag,
   ARC-Easy, ARC-Challenge, PIQA, LAMBADA, OpenBookQA, WinoGrande, or BoolQ.
   Apply before the held-out split is drawn so val is clean too.

7. **Write.** Two artifacts:
   - `raw-mix-v1`: the sampled, filtered text as jsonl.zst shards, one
     directory per source, roughly 8GB.
   - `tok-mix-v1`: uint16 token shards plus per-shard document offset arrays,
     roughly 4GB.

8. **Manifest.** One JSON file: source list with revisions, token counts
   achieved vs target, sampling seeds, tokenizer hash, filter settings, and the
   eval contamination report.

Token-level shards store documents separated by `<|endoftext|>` with a
companion `offsets.npy` per shard. Packing into training sequences happens at
load time, so changing context length later does not require reprocessing.

---

## 5. Mixing during training

Do not concatenate by source. A loader that reads all of FineWeb first spends
the first 670M tokens in one domain, which is a waste for a model this small.

Instead: weighted document interleaving. The dataloader samples documents from
domains with probability proportional to the phase targets, packs them to the
context length, and mixes phase 1 and phase 2 by a step schedule. Phase 2 is
the final 25% of steps, with the LR decay to match.

Per-domain offsets make this cheap. Token-weighted sampling keeps ratios exact
even though document lengths differ wildly between, say, IRC lines and arxiv
papers.

---

## 6. Storage layout

Dataset storage is separate from live session storage, so this is cheap.

**Kaggle Dataset `raw-mix-v1`**, private, about 8GB. Keeps re-tokenization free
if we change the tokenizer or rebalance the mix.

**Kaggle Dataset `tok-mix-v1`**, private, about 4GB:

```
tok-mix-v1/
  train/
    web-000.bin, web-000.offsets.npy
    books-000.bin, books-000.offsets.npy
    ...
  val/
    val-000.bin, val-000.offsets.npy
  tokenizer.json
  manifest.json
```

Mounted read-only at `/kaggle/input/tok-mix-v1` and memmapped by the loader.
The training kernel never tokenizes, never downloads, and does not compete for
the 4 vCPUs.

---

## 7. Open items

No blockers. Decisions still to confirm in the training spec, not this one:
context length, batch token budget, and the phase boundary as a step number
rather than a percentage.
