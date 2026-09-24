#!/usr/bin/env bash
# Full training of every model, sequentially, with per-run logs in results/logs/.
# Usage:  PYTHON=/path/to/python scripts/run_all.sh [model ...]
#   no args  -> cnn efficientnet deit pinn(λ=0.1) pinn(λ=0)  (the λ=0.01 and λ=1 runs are optional extras)
# On macOS the run is wrapped in `caffeinate` so the machine doesn't sleep.
# Finished runs (results/<run>/metrics.json exists) are skipped, so it is safe to re-launch.
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-results}"          # e.g. OUT_DIR=results/smoke EXTRA_ARGS="--train-per-class 50 --epochs 1"
EXTRA="${EXTRA_ARGS:-}"
mkdir -p "$OUT/logs"

JOBS=("$@")
[ ${#JOBS[@]} -eq 0 ] && JOBS=(cnn efficientnet deit pinn:0.1 pinn:0)

run() {
  local job="$1"; [ "$job" = pinn ] && job="pinn:0.1"
  local model="${job%%:*}" lam="" name="$job" args=()
  if [[ "$job" == *:* ]]; then lam="${job#*:}"; name="pinn_lambda${lam}"; args+=(--lambda-poisson "$lam"); fi
  if [ -f "$OUT/$name/metrics.json" ]; then echo "skip $name (done)"; return; fi
  echo "=== $name  $(date '+%F %T')"
  "$PY" scripts/train.py --model "$model" --out-dir "$OUT" ${args[@]+"${args[@]}"} $EXTRA 2>&1 | grep -v HF_TOKEN | tee "$OUT/logs/$name.log"
}

CAFF=""; command -v caffeinate >/dev/null && CAFF="caffeinate -i"
for j in "${JOBS[@]}"; do $CAFF bash -c "$(declare -f run); PY='$PY' OUT='$OUT' EXTRA='$EXTRA'; run '$j'"; done
echo "all done $(date '+%F %T')"
