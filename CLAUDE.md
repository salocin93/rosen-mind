# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MiniMind trains a small (~tens of M params) Chinese/English LLM from scratch in plain PyTorch. It covers tokenizer, pretraining, SFT, LoRA, distillation, DPO, PPO/GRPO/CISPO RL and agentic (tool-calling) RL. `README.md` (Chinese) and `README_en.md` are long and serve as the main docs. Code comments and argparse help strings are mostly in Chinese.

## RosenMind (this fork's English model)

This checkout trains **RosenMind** (identity: "created by Nicolas Rosen"), an English-only model, not the upstream Chinese MiniMind:
- `model/tokenizer.json`/`tokenizer_config.json` are a retrained **English** 6400-vocab BPE (same special-token IDs and chat template as upstream). The original Chinese tokenizer is backed up in `model/tokenizer_zh_backup/`. Weights trained with one tokenizer are useless with the other.
- Data is built by `scripts/prepare_rosenmind.py` from `dataset/raw_en/` (fineweb-edu `sample/10BT/000_00000.parquet`, smol-smoltalk): `pretrain_en.jsonl` (docs pre-split into ≤510-token chunks), `sft_en.jsonl` (smol-smoltalk trimmed to whole turns ≤740 tokens + generated identity conversations).
- `./train.sh` runs both stages with the RosenMind defaults (see `PRETRAIN_ARGS`/`SFT_ARGS` in it), logs to `logs/<timestamp>/`, and sends Uptime Kuma push heartbeats when `KUMA_PUSH_URL` is set. Run it inside tmux.
- 8 GB GPU (RTX 4060 Ti): pretraining at `--max_seq_len 512` fits only up to batch 24; the run uses batch 16 × accumulation 8.
- The Hugging Face hub may hold a stale login token; public datasets download fine with `token=False`.
- **v2** (`v2/`, see `v2/README.md`) is a self-contained second generation (768×14, ~108M; mixed data; packed pre-tokenized uint16 data in `v2/data/bin/`; WSD schedule; synthetic SFT data from the local Qwen3.6-35B via llama-swap; DPO; benchmarks). It has its own copy of the model code and tokenizer, so editing v2 never affects v1. Run stages with `v2/run_v2.sh <stage...>`; `touch v2/STOP` pauses training cleanly.
- Public chat demo: `v2/demo/app.py` (FastAPI + hand-written `static/index.html`, CPU, 127.0.0.1:8000) run by the systemd unit `rosenmind-demo` (`v2/demo/rosenmind-demo.service`); Tailscale Funnel publishes `https://nrosen-server.komodo-cirius.ts.net` → :8000. Keep `huggingface-hub<1.0` (FastAPI/uvicorn came in with `gradio<6`; Gradio 6 pulls hub 2.x, which breaks `transformers` 4.57).
- Never `pkill -f` with a pattern that also appears in your own command line (it kills the calling shell).

## Environment

- Python 3.12, with a uv-managed `.venv`. `pyproject.toml` and `src/minimind/` are a fresh `uv init` stub (they declare no dependencies and are untracked). The real dependencies are in `requirements.txt`, and torch is installed separately (it's commented out there): `uv pip install -r requirements.txt`.
- There are no tests, linter or build step. To check a change, run the relevant script.

## Commands

The training scripts use relative paths (`../dataset`, `../model`, `../out`, `../checkpoints`), so **run them from inside `trainer/`**. Run `eval_llm.py` from the repo root.

```bash
cd trainer && python train_pretrain.py            # -> out/pretrain_768.pth
cd trainer && python train_full_sft.py            # loads out/pretrain_768.pth -> out/full_sft_768.pth
cd trainer && python train_pretrain.py --from_resume 1   # resume from checkpoints/<weight>_<dim>_resume.pth
cd trainer && torchrun --nproc_per_node N train_xxx.py   # multi-GPU DDP (NCCL)
python eval_llm.py --weight full_sft              # chat with out/full_sft_768.pth (prompts for auto/manual mode on stdin)
python eval_llm.py --load_from ./minimind-3       # chat with a transformers-format model dir
cd scripts && streamlit run web_demo.py           # needs a transformers-format model folder copied into scripts/
```

Shared flags across trainers: `--hidden_size` (default 768), `--num_hidden_layers` (8), `--use_moe 0|1`, `--from_weight`, `--save_weight`, `--data_path`, `--max_seq_len`, `--use_wandb` (it actually imports `swanlab as wandb`), `--use_compile`. For a quick smoke test on CPU, pass small `--batch_size`/`--max_seq_len` and `--device cpu`.

Datasets go in `dataset/` (download from ModelScope `gongjy/minimind_dataset`). The upstream Chinese datasets were deleted; only the English `pretrain_en.jsonl`/`sft_en.jsonl` are present. They are multi-GB, so never read them whole. Use `head`.

## Architecture

**Weight-naming convention.** Everything relies on it. Weights are saved and loaded as `{save_dir}/{weight}_{hidden_size}{_moe?}.pth`, as fp16 state dicts. Each stage's `--from_weight` names the previous stage's `--save_weight`: `pretrain` → `full_sft` → (`dpo`, `ppo_actor`, `grpo`, `agent`, `lora_*`, `full_dist`). Resume checkpoints (model + optimizer + scaler + step + world_size + wandb id) go to `checkpoints/{weight}_{dim}{_moe}_resume.pth`. On resume, the step is rescaled if the GPU count changed. Model architecture flags (`hidden_size`, `num_hidden_layers`, `use_moe`) must match between saving and loading.

**Model (`model/model_minimind.py`).** A single file containing `MiniMindConfig` (a HF `PretrainedConfig`) and `MiniMindForCausalLM` (a HF `PreTrainedModel` + `GenerationMixin`). It's a Llama/Qwen3-style decoder: RMSNorm, GQA (8 query heads / 4 KV heads) with q/k-norm, RoPE (θ=1e6) with optional YaRN scaling, SwiGLU FFN, and tied embeddings. The vocab is 6400 (custom tokenizer in `model/tokenizer.json`). The optional MoE FFN (`MOEFeedForward`) adds `aux_loss` to the output, and every trainer adds `res.loss + res.aux_loss`. `generate()` is custom (it overrides HF's) and uses a tuple-list KV cache, not HF `Cache` objects. Loss shifting happens inside `forward`, so labels equal input_ids, with -100 marking positions that are ignored.

**Training scripts (`trainer/`).** Each `train_*.py` is a standalone script with the same skeleton: argparse → `init_distributed_mode` → `lm_checkpoint` (load resume) → autocast/GradScaler → `init_model` → Dataset + `SkipBatchSampler` (skips already-seen batches on resume) → `train_epoch` loop with a cosine LR (`get_lr`), grad accumulation and periodic saves. Shared helpers are in `trainer/trainer_utils.py`. Scripts do `sys.path.append(..)` so that `model.` / `dataset.` / `trainer.` imports work from inside `trainer/`.
- RL trainers (`train_ppo.py`, `train_grpo.py`, `train_agent.py`) need an external reward model (`--reward_model_path`, default `../../internlm2-1_8b-reward`, which lives outside the repo). Their rollouts go through `trainer/rollout_engine.py` (`--rollout_engine torch|sglang`). sglang requires a separately launched server serving a transformers-format model.
- `train_agent.py` does multi-turn tool-calling RL. `safe_math_eval` in `trainer_utils.py` is the sandboxed calculator tool, so don't swap it for `eval`.

**Datasets (`dataset/lm_dataset.py`).** One class per stage: `PretrainDataset` (raw `text`, BOS/EOS-wrapped and padded), `SFTDataset` (`conversations` rendered through the tokenizer's chat template), `DPODataset`, `RLAIFDataset` and `AgentRLDataset`. SFT labels are masked so that only assistant spans are trained on: the code finds the `<bos>assistant\n` … `<eos>\n` token sequences. Changing the chat template in `tokenizer_config.json` therefore affects label masking. SFT preprocessing randomly injects system prompts (20%) and strips empty `<think>` blocks (80%). Tool-use samples are left untouched.

**LoRA (`model/model_lora.py`).** Monkey-patches the `forward` of square `nn.Linear` layers. LoRA weights are saved separately and applied at inference with `eval_llm.py --lora_weight <name>`, and `--lora_rank` must match the rank used in training.

**Conversion and serving (`scripts/`).** `convert_model.py` converts native `.pth` files to and from the HF transformers format and merges LoRA weights. `serve_openai_api.py` provides an OpenAI-compatible API. `eval_toolcall.py` tests tool calling.
