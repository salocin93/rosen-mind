#!/usr/bin/env bash
# RosenMind v2 pipeline. Runs the given stages in order, stops at the first failure, sends Uptime Kuma heartbeats.
#
#   KUMA_PUSH_URL='http://nrosen-pi:3001/api/push/<token>' v2/run_v2.sh bench mix pretrain
#   v2/run_v2.sh sft-data sft dpo-data dpo eval
#
# Stages:
#   bench     GPU speed/memory benchmark -> picks batch size / compile           (~10 min, needs a free GPU)
#   mix       pick pretraining rows for HOURS (default 24) of training             (~1 min)
#   pretrain  v2/train.py pretrain                                                 (~HOURS)
#   synth     35B synthetic SFT data via llama-swap (starts/stops the container)   (~1-3 h, needs a free GPU)
#   sft-data  pack SFT data (smol-smoltalk + identity + synthetic)                 (~5 min)
#   sft       v2/train.py sft
#   dpo-data  build DPO pairs                                                      (~3 min)
#   dpo       v2/train_dpo.py
#   eval      benchmarks + chat samples for v1 and every v2 stage
# Rerunning a training stage resumes it. `touch v2/STOP` pauses training cleanly (delete it before resuming).
set -uo pipefail
V2="$(dirname "$(readlink -f "$0")")"; ROOT="$(dirname "$V2")"
cd "$ROOT" && source .venv/bin/activate
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$V2/logs"
HOURS="${HOURS:-24}"
STAGE_LOG=""

push() {
  [[ -z "${KUMA_PUSH_URL:-}" ]] && return 0
  curl -fsS -m 10 -G "${KUMA_PUSH_URL%%\?*}" --data-urlencode "status=$1" --data-urlencode "msg=${2:0:250}" --data-urlencode "ping=" >/dev/null 2>&1 || true
}
heartbeat() {
  while sleep 60; do
    [[ -f "$V2/.stage" ]] || continue
    local stage log line gpu age
    read -r stage log < "$V2/.stage"
    line=$(grep -aE '(step |val |call )' "$log" 2>/dev/null | tail -1 | cut -c1-140)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null)
    age=$(( ($(date +%s) - $(stat -c %Y "$log" 2>/dev/null || date +%s)) / 60 ))
    if (( age >= ${STALL_MIN:-20} )); then push down "v2 $stage: no output for ${age} min | GPU $gpu"
    else push up "v2 $stage | ${line:-starting} | GPU $gpu"; fi
  done
}
bench_val() { python -c "import json;b=json.load(open('$V2/data/bin/gpu_bench.json'))['$1'];print(b['$2'])"; }

run() {  # run <stage> <command...>
  local stage=$1; shift
  STAGE_LOG="$V2/logs/run_${stage}.log"
  echo "$stage $STAGE_LOG" > "$V2/.stage"
  echo "=== $(date '+%F %T') $stage: $*" | tee -a "$STAGE_LOG"
  push up "v2 $stage: starting"
  "$@" 2>&1 | tee -a "$STAGE_LOG"
  local rc=${PIPESTATUS[0]}
  if (( rc != 0 )); then
    push down "v2 $stage FAILED (exit $rc): $(grep -av '^\s*$' "$STAGE_LOG" | tail -1 | cut -c1-200)"
    echo "=== $stage failed (exit $rc)"; exit $rc
  fi
  if grep -aq 'STOP file found' "$STAGE_LOG" 2>/dev/null && [[ -f "$V2/STOP" ]]; then
    push up "v2 $stage paused by STOP file"; echo "=== paused by STOP file"; exit 0
  fi
  echo "=== $(date '+%F %T') $stage done"
}

stage() {
  case "$1" in
    bench)    run bench python "$V2/bench_gpu.py" ;;
    mix)      run mix python "$V2/scripts/build_pretrain.py" mix --hours "$HOURS" ;;
    pretrain) run pretrain python "$V2/train.py" pretrain --batch_size "$(bench_val best batch_size)" \
                --accum "$(bench_val best accum)" --compile "$(bench_val best compile)" ;;
    synth)
      docker start llama-swap >/dev/null
      export PC_API_KEY=$(docker inspect llama-swap --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^PC_API_KEY=//p')
      run synth-identity python "$V2/scripts/gen_synthetic.py" identity --calls "${SYNTH_IDENTITY_CALLS:-150}"
      run synth-general python "$V2/scripts/gen_synthetic.py" general --calls "${SYNTH_GENERAL_CALLS:-300}"
      docker stop llama-swap >/dev/null ;;
    sft-data) run sft-data nice -n 10 python "$V2/scripts/build_sft.py" ;;
    sft)      run sft python "$V2/train.py" sft --batch_size "$(bench_val best_sft batch_size)" --accum "$(bench_val best_sft accum)" ;;
    dpo-data) run dpo-data nice -n 10 python "$V2/scripts/build_dpo.py" ;;
    dpo)      run dpo python "$V2/train_dpo.py" ;;
    eval)
      for w in pretrain_768 full_sft_768; do
        [[ -f out/$w.pth ]] && run "eval-v1-$w" python "$V2/eval/bench.py" --weights "out/$w.pth" --arch v1
      done
      for w in pretrain_v2 sft_v2 dpo_v2; do
        [[ -f "$V2/out/$w.pth" ]] && run "eval-$w" python "$V2/eval/bench.py" --weights "$V2/out/$w.pth"
      done
      [[ -f "$V2/out/sft_v2.pth" ]] && run chat-sft python "$V2/eval/chat.py" --weights "$V2/out/sft_v2.pth"
      [[ -f "$V2/out/dpo_v2.pth" ]] && run chat-dpo python "$V2/eval/chat.py" --weights "$V2/out/dpo_v2.pth"
      python "$V2/eval/compare.py" | tee "$V2/logs/compare.txt" ;;
    *) echo "unknown stage: $1"; exit 2 ;;
  esac
}

(( $# )) || { sed -n '2,20p' "$0"; exit 2; }
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null; rm -f "$V2/.stage"' EXIT
trap 'push down "v2 interrupted"; exit 130' INT TERM HUP
for s in "$@"; do stage "$s"; done
push up "v2 stages done: $*"
echo "=== all requested stages done: $*"
