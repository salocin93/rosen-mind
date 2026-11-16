#!/usr/bin/env bash
# Runs pretrain -> full SFT, logging to logs/<timestamp>/, with optional Uptime Kuma push heartbeats.
#
# Usage:
#   KUMA_PUSH_URL='https://kuma.example.com/api/push/XXXXXXXX' ./train.sh
# Env options:
#   START=sft     skip pretraining (needs out/pretrain_768.pth)
#   RESUME=1      pass --from_resume 1 to the training scripts (continue from checkpoints/)
#   STALL_MIN=15  report "down" if the log hasn't changed for this many minutes
#   PRETRAIN_ARGS / SFT_ARGS  extra arguments for each stage (word-split), e.g. PRETRAIN_ARGS="--epochs 1"
#   Defaults below are the RosenMind English run (see scripts/prepare_rosenmind.py).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source .venv/bin/activate
export PYTHONUNBUFFERED=1  # otherwise python block-buffers stdout into tee and the log/heartbeat see nothing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PRETRAIN_ARGS="${PRETRAIN_ARGS---data_path ../dataset/pretrain_en.jsonl --max_seq_len 512 --batch_size 16 --accumulation_steps 8 --epochs 1 --num_workers 6}"
SFT_ARGS="${SFT_ARGS---data_path ../dataset/sft_en.jsonl --epochs 1 --learning_rate 5e-5 --num_workers 6}"

RUN_DIR="logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
STAGE_FILE="$RUN_DIR/.stage"
STALL_MIN="${STALL_MIN:-15}"
EXTRA=()
[[ "${RESUME:-0}" == 1 ]] && EXTRA+=(--from_resume 1)

push() {  # push <up|down> <message>
  [[ -z "${KUMA_PUSH_URL:-}" ]] && return 0
  curl -fsS -m 10 -G "${KUMA_PUSH_URL%%\?*}" \
    --data-urlencode "status=$1" --data-urlencode "msg=${2:0:250}" --data-urlencode "ping=" >/dev/null 2>&1 || true
}

heartbeat() {
  while sleep 60; do
    local stage log line gpu age
    stage=$(cat "$STAGE_FILE" 2>/dev/null) || continue
    log="$RUN_DIR/$stage.log"
    line=$(grep -a 'Epoch:' "$log" 2>/dev/null | tail -1 | sed 's/, logits_loss.*lr:/, lr:/')
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)
    age=$(( ($(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s)) / 60 ))
    if (( age >= STALL_MIN )); then
      push down "$stage: no log output for ${age} min (stalled?) | GPU $gpu"
    else
      push up "$stage | ${line:-loading data} | GPU $gpu"
    fi
  done
}

run_stage() {  # run_stage <name> <script> <extra args string>
  echo "$1" > "$STAGE_FILE"
  echo "=== $(date '+%F %T') starting $1 -> $RUN_DIR/$1.log  (args: $3 ${EXTRA[*]})"
  push up "$1: starting"
  # shellcheck disable=SC2086
  (cd trainer && python "$2" $3 "${EXTRA[@]}") 2>&1 | tee "$RUN_DIR/$1.log"
  local rc=$?
  if (( rc != 0 )); then
    push down "$1 FAILED (exit $rc): $(grep -av '^\s*$' "$RUN_DIR/$1.log" | tail -1)"
    echo "=== $(date '+%F %T') $1 failed with exit code $rc"
    exit $rc
  fi
  echo "=== $(date '+%F %T') finished $1"
}

heartbeat & HB_PID=$!
trap 'kill $HB_PID 2>/dev/null' EXIT
trap 'push down "training interrupted"; exit 130' INT TERM HUP

[[ "${START:-pretrain}" == pretrain ]] && run_stage pretrain train_pretrain.py "$PRETRAIN_ARGS"
run_stage sft train_full_sft.py "$SFT_ARGS"

kill $HB_PID 2>/dev/null
push up "ALL DONE: out/full_sft_768.pth ready"
echo "=== $(date '+%F %T') all stages done. Try: python eval_llm.py --weight full_sft"
