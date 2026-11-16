# RosenMind v2

Second-generation RosenMind ("created by Nicolas Rosen"), independent of the v1 code in the repo root
(v2 has its own copy of the model code and tokenizer, so v1 can keep training/serving untouched).

| | v1 | v2 |
|---|---|---|
| Model | 768 × 8 layers, 64M | 768 × **14 layers, ~108M** |
| Tokenizer | English BPE 6400 | same (val losses directly comparable) |
| Pretraining data | fineweb-edu, 820M tokens | **~2B tokens**: fineweb-edu 70 / cosmopedia-v2 15 / finemath 10 / Python 5 |
| Schedule | cosine, no warmup | **warmup → flat → decay**, decay during a final **quality phase** (fineweb-edu score ≥ 4, more textbook/math/code) |
| Data loading | jsonl, re-tokenized, padded | pre-tokenized, **packed** (no padding), in pretraining and SFT |
| SFT | smol-smoltalk + templated identity | + **synthetic data from local Qwen3.6-35B**, packed 1024-token rows |
| Alignment | – | **DPO** (ultrafeedback + identity preference pairs) |
| Evaluation | – | val loss per source, HellaSwag / ARC-E / ARC-C / PIQA, chat samples |

## Run order

```bash
cd ~/RosenMind
export KUMA_PUSH_URL='http://nrosen-pi:3001/api/push/<token>'     # optional heartbeats
# data that is already prepared: v2/data/bin/pool_*.bin (pretraining pools)

# 1. after v1 has finished (needs the whole GPU):
tmux new -s v2 "v2/run_v2.sh bench mix pretrain; exec bash"       # ~10 min + 1 min + ~24 h
# 2. after pretraining:
tmux new -s v2post "v2/run_v2.sh synth sft-data sft dpo-data dpo eval; exec bash"
```

Each stage can be run on its own (`v2/run_v2.sh eval`). Training stages resume when rerun.
`touch v2/STOP` pauses the current training stage cleanly (it saves first); `rm v2/STOP` and rerun to continue.
Logs: `v2/logs/run_<stage>.log` (+ `<name>.log` / `<name>.csv` metrics from the trainers).

## Files

- `common.py` – paths, architecture (`MODEL_V2`), LR schedule, optimizer, helpers
- `scripts/build_pretrain.py` – `tokenize` (raw parquet → packed pools) and `mix` (token budget → row order)
- `scripts/gen_synthetic.py` – identity + general Q&A from the local 35B via llama-swap
- `scripts/identity.py` – templated identity conversations and DPO identity pairs
- `scripts/build_sft.py`, `scripts/build_dpo.py` – packed SFT rows / DPO pairs
- `train.py` – pretraining + SFT; `train_dpo.py` – DPO
- `bench_gpu.py` – picks batch size / torch.compile for this GPU
- `eval/bench.py` (benchmarks), `eval/chat.py` (spot checks / interactive), `eval/compare.py` (table)

Raw data sources (in `data/raw/`, plus fineweb shard 000 and smol-smoltalk reused from v1's `dataset/raw_en/`):
HuggingFaceFW/fineweb-edu sample-10BT (000, 001), HuggingFaceTB/smollm-corpus cosmopedia-v2 (2 shards),
HuggingFaceTB/finemath finemath-4plus (3 shards), Nan-Do/code-search-net-python, HuggingFaceH4/ultrafeedback_binarized,
eval sets Rowan/hellaswag, allenai/ai2_arc, baber/piqa.
