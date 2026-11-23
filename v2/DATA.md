# RosenMind v2 data setup

This document describes the data for every RosenMind v2 training stage: where each dataset comes from, how it is processed, what lands on disk and how the trainers read it. Figures are as of 2026-11-17, after `run_v2.sh synth sft-data dpo-data` and the pool rebuild that added python-edu and raised the caps. Pretraining has **not** run yet: `v2/out/` and `v2/checkpoints/` are empty, and `data/bin/gpu_bench.json` does not exist yet.

## 1. Overview

| Stage | Built by | Files read by the trainer | Size | Read by |
|---|---|---|---|---|
| Pretraining | `scripts/build_pretrain.py tokenize` + `mix` | `pool_*.bin`, `order.npy`, `order.json` | 4.86B tokens in the current mix (5.35B in the pools) | `train.py pretrain` |
| SFT | `scripts/gen_synthetic.py` + `scripts/build_sft.py` | `sft_ids.bin`, `sft_mask.bin`, `sft.json` | 288,686 rows × 1024 = 296M tokens (216M trained on) | `train.py sft` |
| DPO | `scripts/build_dpo.py` | `dpo_train.npz`, `dpo_val.npz` | 20,550 + 462 pairs | `train_dpo.py` |
| Evaluation | (raw parquet) | `data/raw/{Rowan__hellaswag, allenai__ai2_arc, baber__piqa}` and the held-out rows of each pretraining pool | – | `eval/bench.py` |

All tokens use the v2 copy of the English 6400-vocab BPE tokenizer (`v2/tokenizer/`). It is identical to v1's, so validation losses can be compared directly between v1 and v2. Token IDs fit in `uint16`. The special IDs are `PAD=0` (`<|endoftext|>`), `BOS=1` (`<|im_start|>`) and `EOS=2` (`<|im_end|>`), defined in `common.py`.

```
v2/data/
├── raw/          downloaded Hugging Face parquet files (~10 GB)
├── synthetic/    35B-generated conversations (identity.jsonl, general.jsonl + .done resume markers)
└── bin/          everything the trainers read (pre-tokenized, packed)
    ├── pool_{fineweb,fineweb_hq,cosmo,math,code}.bin   pools.json
    ├── order.npy  order.json                           pretraining row order
    ├── sft_ids.bin  sft_mask.bin  sft.json             SFT
    └── dpo_train.npz  dpo_val.npz                      DPO
```

`v2/data/` is not tracked by git (it is untracked in `git status`).

## 2. Raw sources

| Source (HF id) | Files | Used for |
|---|---|---|
| `HuggingFaceFW/fineweb-edu`, `sample/10BT` | shard `000_00000` (reused from v1's `dataset/raw_en/sample/`), plus `001_00000`, `002_00000` and `003_00000` in `data/raw/` | pretraining: `fineweb` and `fineweb_hq` |
| `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 2 of 104 shards | pretraining: `cosmo` (synthetic textbooks) |
| `HuggingFaceTB/finemath`, `finemath-4plus` | 3 of 64 shards | pretraining: `math` |
| `Nan-Do/code-search-net-python` | all 4 shards (the `original_string` column, i.e. function source with docstrings); no more exist | pretraining: `code` |
| `Avelina/python-edu-cleaned` (an ungated copy of SmolLM's Python-Edu: whole GitHub Python files with an educational score) | 1 of 27 shards (`train-00000`, 284k files, every one scoring 4–5) | pretraining: `code` |
| `HuggingFaceTB/smol-smoltalk` | v1's `dataset/raw_en/data/train-*.parquet` | SFT |
| `HuggingFaceH4/ultrafeedback_binarized` | `train_prefs`, `test_prefs` | DPO |
| `Rowan/hellaswag` (val), `allenai/ai2_arc` (ARC-Easy and ARC-Challenge test sets), `baber/piqa` (val) | – | benchmarks only, never trained on |

## 3. Pretraining data

### 3.1 Tokenize: raw parquet → packed pools

`build_pretrain.py tokenize` streams each source and writes one flat `uint16` stream per pool:

```
[BOS] doc tokens [EOS] [BOS] doc tokens [EOS] ...
```

The stream is then truncated to a multiple of 512 tokens, so every pool is a matrix of 512-token rows with **no padding at all**. Documents are not split or aligned to row boundaries, so a row usually contains the end of one document and the start of the next, separated by `EOS BOS`. Each source stops at a token cap.

Fineweb-edu is split by its `int_score` classifier score: documents scoring 4 or higher go to `fineweb_hq` until that pool reaches its cap, and everything else (including surplus score-4+ documents) goes to `fineweb`.

| Pool | Cap | Docs | Tokens | 512-token rows | Content |
|---|---|---|---|---|---|
| `fineweb` | 3.30B | 2,600,481 | 3,300M | 6,446,104 | fineweb-edu, score < 4 (plus surplus score ≥ 4 docs) |
| `fineweb_hq` | 0.40B | 297,519 | 400M | 781,368 | fineweb-edu, score ≥ 4 |
| `cosmo` | 0.80B | 752,578 (all) | 709M | 1,383,934 | Cosmopedia v2 synthetic textbooks/stories |
| `math` | 0.65B | 314,040 (all) | 586M | 1,144,482 | FineMath 4+ |
| `code` | 0.45B | 739,594 (all) | 356M | 695,783 | 455k CodeSearchNet Python functions, then 284k python-edu files |
| **total** | | | **5.35B** | 10,451,671 | |

The `cosmo`, `math` and `code` caps are above what their downloaded shards hold, so those pools contain every document; adding shards grows them. Fineweb is capped: its four shards hold 2.92M documents, about 3.7B tokens, and the two pools use 2.90M of those documents. In code, Windows line endings (`\r\n`, in 14% of python-edu files and 3% of CodeSearchNet functions) are normalized to `\n` before tokenizing. The code pool streams CodeSearchNet first and python-edu last, so its validation rows (the tail of the pool) are python-edu files.

The counts come from `data/bin/pools.json`. A pool can be rebuilt by itself with `--sources fineweb,fineweb_hq`, which is how the third fineweb shard was added; the other pools are left untouched.

**Validation hold-out.** The last 256 rows of each pool (`VAL_ROWS`) are never put into the training order. `train.py` reports validation loss on each source separately, and `eval/bench.py` uses the same rows.

### 3.2 Mix: token budget → row order

`build_pretrain.py mix` chooses which rows are trained on and in what order, and writes:

- `order.npy`: an `int64` array of `(pool index, row index)` pairs, one per training row.
- `order.json`: the metadata (`sources`, `seq`, `rows`, `tokens`, `hq_start`, per-phase row counts, `val_rows`).

The budget is either `--tokens N` or `--hours H`. With `--hours`, the budget is benchmark tokens/s (from `gpu_bench.json`) × 3600 × H × 0.95. `run_v2.sh mix` uses `--hours $HOURS` (48 by default), so **running `bench mix` overwrites the current `order.npy` with one sized to the measured GPU speed**.

The training run has two phases:

| Phase | Planned share | Planned mix |
|---|---|---|
| Main (shuffled) | 85% | fineweb 70 / cosmo 15 / math 10 / code 5 |
| Quality (last, while the LR decays) | 15% | fineweb_hq 50 / cosmo 20 / math 20 / code 10 |

Every pool is permuted once, and the two phases take disjoint slices of it, so **no row is ever seen twice**. When a small pool cannot cover its planned share, the missing rows come from fineweb: first by taking that pool's quality-phase rows away (and giving them to `fineweb_hq`), then by taking main-phase rows away. If `fineweb_hq` then runs out too, the leftover moves to main-phase `fineweb`. Each phase is shuffled separately, the quality phase is appended after the main phase, and `hq_start` records where it begins.

**The order currently on disk** (`order.json`) was built by `run_v2.sh mix` from the GPU benchmark. At 29,591 tokens/s × 48 h × 0.95 the budget is **4.86B tokens** (9,487,611 rows, 74,122 optimizer steps of 65,536 tokens):

| Phase | Rows | Tokens | fineweb | fineweb_hq | cosmo | math | code |
|---|---|---|---|---|---|---|---|
| Main | 8,105,549 | 4.15B | 5,686,209 (70.2%) | – | 1,209,670 (14.9%) | 806,447 (9.9%) | 403,223 (5.0%) |
| Quality | 1,382,062 | 0.71B | – | 781,112 (56.5%) | 174,008 (12.6%) | 284,628 (20.6%) | 142,314 (10.3%) |

Cosmopedia is 110,620 rows short, so its quality-phase share drops from 20% to 12.6%. Those rows go to `fineweb_hq`, which is then used up completely, so the last 41k overflow rows go to main-phase fineweb. The quality phase (and the LR decay) is 14.6% of training. All of cosmo, fineweb_hq, math and code are used. Fineweb has about 0.76M unused rows, so the mix holds up to roughly 5.3B tokens. Beyond that `mix` stops with "not enough fineweb rows", and you need another fineweb-edu shard (and a higher cap).

### 3.3 How the trainer reads it

`train.py pretrain` memory-maps the pools (`open_rows`). For each training index it reads `order[i] = (src, row)` and trains on that row with `labels = input_ids`; the shift happens inside the model. Resuming continues from an offset (`FromOffset` sampler), so the order is deterministic. The LR schedule is warmup-stable-decay (`common.wsd_lr`): 1% linear warmup, then a flat peak of 6e-4, then cosine decay down to 10% of the peak. The decay begins at `hq_start / rows`, so it coincides exactly with the quality phase.

## 4. SFT data

### 4.1 Sources

| Source | Conversations | Weight | Notes |
|---|---|---|---|
| smol-smoltalk | 460,341 | ×1 | multi-turn general chat (the same data as v1) |
| Templated identity (`scripts/identity.py`) | 3,000 | ×1 | 52 fixed questions (plus lowercase variants), fixed answers from `WHO`/`SPECIFIC`, 20% with a system prompt, 25% with a short follow-up turn |
| `synthetic/identity.jsonl` | 1,499 | ×2 | 35B-written identity Q&A |
| `synthetic/general.jsonl` | 2,303 | ×2 | 35B-written short Q&A over 88 topics |

Identity conversations make up about 6,000 of 467k (1.3%).

### 4.2 Synthetic generation (`gen_synthetic.py`)

- **Model:** local Qwen3.6-35B-A3B served by the `llama-swap` Docker container (OpenAI-compatible API on `:9292`, key `PC_API_KEY` read from the container's environment), with thinking disabled, temperature 0.8, 4 parallel workers and up to 3 retries per call. `run_v2.sh synth` starts and stops the container itself.
- **Identity task:** 150 calls × 10 conversations. Each prompt includes a fixed `FACTS` block (name, version 2, creator Nicolas Rosen as an individual, about 152M parameters, RTX 4060 Ti, MiniMind code, training data, no internet/memory, "not ChatGPT/Claude/Llama/Qwen/Gemini/MiniMind") and one random user persona out of 12 (child, skeptic, typo-heavy, "testing whether it is ChatGPT", …). A filter drops any reply that names OpenAI/Anthropic/Alibaba/Meta/Google/DeepMind/jingyaogong without a negation, so "not made by OpenAI" passes and "made by OpenAI" is dropped.
- **General task:** 300 calls × 8 conversations, each call on one random topic. The prompt asks for correct, simple English in 2–6 sentences or a short numbered list, with no headings, tables or disclaimers, and mixed question types.
- **Parsing:** strips code fences, takes the outermost JSON array, and keeps conversations that start with the user, end with the assistant and alternate roles.
- **Resume:** each finished call ID is appended to `<task>.done`, and reruns skip those calls. Rerunning with a higher `--calls` adds more data.
- **Last run:** identity took 35 min (all 150 calls OK) and general took 85 min (all 300 calls OK).

Checks on the outputs: 0 empty messages, 0 `<think>` tags, 0 replies claiming to be another model, and 0 exact duplicates in the general set. Known flaws are listed in §7.

### 4.3 Building the packed rows (`build_sft.py`)

1. All conversations are loaded, synthetic ones twice, and shuffled with seed 0.
2. **System prompts:** 20% of conversations that lack one get a random prompt from `identity.SYSTEM_PROMPTS`, two of which name RosenMind/Nicolas Rosen.
3. **Trimming:** each conversation is rendered with the chat template, keeping the longest run of whole turns that ends on an assistant reply and fits in 1024 tokens. If even the first exchange doesn't fit, the conversation is dropped (11,699 dropped).
4. **Loss mask:** 1 on every assistant-reply token from after `<|im_start|>assistant\n` up to and including `<|im_end|>\n`, and 0 elsewhere (system and user turns, template tokens, padding).
5. **Best-fit packing:** up to 64 rows stay open. Each conversation goes into the open row where it fits most tightly, or a new row is opened (retiring the fullest row when 64 are already open). A row is closed once fewer than 16 tokens are left. The leftover space is filled with `PAD`.

Output (`sft.json`): 459,246 conversations in 288,686 rows of 1024 tokens (296M tokens). Rows are 94.7% full, and 216M tokens (73%) are trained on. `sft_ids.bin` is `uint16` and `sft_mask.bin` is `uint8`; both are flat and are read as `(rows, 1024)` memmaps.

Several conversations share a row without an attention mask between them. They are separated only by the template's `<|im_start|>`/`<|im_end|>` tokens, the same way pretraining rows mix documents.

### 4.4 How the trainer reads it

`train.py sft` takes a random 400 rows (seed 0) as validation and trains on the rest for `--epochs` (default 1), shuffled. Labels are `input_ids` with `mask == 0` positions set to −100. The LR defaults are a 5e-5 peak, 3% warmup and cosine decay. Since rows are packed and synthetic conversations appear twice, a validation row can contain the same conversation as a training row. The SFT validation loss is therefore slightly optimistic, which matters little.

## 5. DPO data (`build_dpo.py`)

- **UltraFeedback:** `train_prefs` pairs are kept only if `score_chosen − score_rejected ≥ 1.0` and the two final replies differ. They are shuffled (seed 0), rendered with the chat template, dropped if either side exceeds 1024 tokens, and capped at `--max_pairs 20000`. Validation uses the first 500 filtered `test_prefs` pairs, of which 412 fit.
- **Identity pairs:** 600 pairs from `identity.preference_pairs`. The prompt is a "who are you" or "are you X" question, the chosen reply is RosenMind's answer, and the rejected reply is a wrong identity ("I am ChatGPT…", "I'm Claude…", "I'm Qwen…", "I am MiniMind… developed by jingyaogong", …). 550 go to training and 50 to validation.
- **Format:** `dpo_{train,val}.npz` with `chosen_ids`/`rejected_ids` (`uint16`, `[n, 1024]`, PAD-filled), `chosen_mask`/`rejected_mask` (`uint8`, assistant-reply tokens only) and `lengths` (the longer of the two sides, so batches can be cropped).
- **Result:** 20,550 training pairs (mean length 492) and 462 validation pairs (mean length 453).
- **Trainer defaults (`train_dpo.py`):** starts from `out/sft_v2.pth`, LR 2e-7, β 0.1, 1 epoch, 4 pairs × 4 accumulation, validation every 200 steps.

## 6. Rebuilding

```bash
cd ~/RosenMind
v2/run_v2.sh bench mix          # GPU benchmark, then a new order.npy sized for $HOURS (default 48)
v2/run_v2.sh synth              # regenerate/extend synthetic data (needs the free GPU; resumable)
v2/run_v2.sh sft-data dpo-data  # repack SFT rows and DPO pairs (~5 + ~3 min, CPU)

# pools (CPU only, safe during training):
RAYON_NUM_THREADS=6 nice -n 19 python v2/scripts/build_pretrain.py tokenize --sources cosmo,math   # default: all pools
python v2/scripts/build_pretrain.py mix --tokens 3.2e9
```

Re-tokenizing all five pools takes about 18 minutes on the 16 CPU threads. The tokenizer is CPU-only, so the GPU does not help.

`build_sft.py` and `build_dpo.py` are deterministic (fixed seeds), so rebuilding without source changes produces the same files. To start a synthetic task from scratch, delete `data/synthetic/<task>.jsonl` and `<task>.done`.

## 7. Known data issues

1. **One stray Chinese word.** One general synthetic answer (electric vs. gas cars) contains "尾气" mid-sentence. Separately, 588 smol-smoltalk conversations (0.13%) contain CJK text, mostly translation requests. Nothing filters either.
2. **Repeated identity data.** The templated set spreads 3,000 conversations over about 52 questions with fixed answers, so each question appears about 58 times. The synthetic identity set repeats some questions heavily ("Are you ChatGPT?" 56×, "Are you ChatGPT or Claude?" 45×, "Who made you?" 40×) and contains 62 exact duplicate question-and-answer pairs, all of which count twice. "Are you ChatGPT?" therefore appears about 160 times in total (49 templated + 56 × 2 synthetic).
