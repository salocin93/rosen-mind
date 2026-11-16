# RosenMind

RosenMind is a small English language model created by Nicolas Rosen and trained from scratch on a single
NVIDIA RTX 4060 Ti (8 GB): its own tokenizer, pretraining, chat fine-tuning and preference tuning.

Try it: **https://nrosen-server.komodo-cirius.ts.net** (it is tiny, so it is fluent but often wrong).

| | v1 (live in the demo) | v2 (prepared) |
|---|---|---|
| Model | 768 × 8 layers, 64M parameters | 768 × 20 layers, 152M parameters |
| Pretraining | FineWeb-Edu, 0.8B tokens, ~6 h | FineWeb-Edu / Cosmopedia / FineMath / Python, ~2.8B tokens, 48 h |
| Fine-tuning | smol-smoltalk + identity conversations | + synthetic conversations from a local Qwen3.6-35B, packed rows |
| Alignment | – | DPO (UltraFeedback + identity pairs) |

## Layout

| Path | What |
|---|---|
| `model/` | model code (`model_minimind.py`) and the English 6400-token tokenizer |
| `trainer/` | v1 training: `train_pretrain.py`, `train_full_sft.py`, `train_tokenizer.py`, shared `trainer_utils.py` |
| `dataset/lm_dataset.py` | v1 dataset classes (pretraining text, chat conversations) |
| `scripts/prepare_rosenmind.py` | v1 data preparation (English corpus, SFT data, identity conversations) |
| `train.sh` | runs v1 pretraining + fine-tuning with Uptime Kuma heartbeats |
| `eval_llm.py` | chat with v1 weights in the terminal |
| `v2/` | the second generation: data pipeline, trainer, DPO, benchmarks, synthetic data, public demo (see `v2/README.md`) |

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu132   # this machine: torch 2.15 + CUDA 13.2; pick yours
uv pip install -r requirements.txt

python v2/eval/chat.py --weights out/full_sft_768.pth --arch v1 -i        # chat with v1
v2/run_v2.sh bench mix pretrain                                          # train v2 (see v2/README.md)
```

Datasets, weights and logs are not in git (see `.gitignore`); `scripts/prepare_rosenmind.py` and
`v2/scripts/` download and build them.

## Credits

RosenMind is built on [MiniMind](https://github.com/jingyaogong/minimind) by Jingyao Gong (Apache-2.0):
the model architecture, chat template and v1 training scripts come from MiniMind. This repository keeps
MiniMind's history and license (`LICENSE`); RosenMind's tokenizer, data, v2 pipeline and demo are new.
