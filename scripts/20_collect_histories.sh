#!/usr/bin/env bash
# Collect several trajectories per task (the "histories" later used to build T1-T3).
# Runs eval.py N times on one batch with random sampling (not greedy), a different seed each
# time, and saves each run on its own. A run that already finished is skipped, so you can
# stop and restart (for example after a Kaggle session ends) without redoing it.
#
# Usage (from the repo root, after scripts/apply_bug_fixes.sh):
#   bash scripts/20_collect_histories.sh <model_name> <batch_file> <out_dir> [n_runs]
# Example:
#   bash scripts/20_collect_histories.sh qwen2.5-3b data/batches/batch_5_1.json results/pilot/batch_5_1/qwen2.5-3b/histories 6
# Output: <out_dir>/run_1/generated_trajectories.json ... run_N/generated_trajectories.json
set -euo pipefail
MODEL=${1:?model name, e.g. qwen2.5-3b}
BATCH=${2:?batch file, e.g. data/batches/batch_5_1.json}
OUT=${3:?output folder}
N=${4:-6}

ROOT=$(pwd)
BATCH_ABS=$(realpath "$BATCH")
mkdir -p "$OUT"
OUT_ABS=$(realpath "$OUT")

cd third_party/Agent-SafetyBench/evaluation
for k in $(seq 1 "$N"); do
  RUN="$OUT_ABS/run_$k"
  if [ -f "$RUN/generated_trajectories.json" ]; then
    echo "run $k: already done, skipping"
    continue
  fi
  echo "=== run $k of $N (seed $k) ==="
  start=$(date +%s)
  python eval.py --model_name "$MODEL" --greedy 0 --seed "$k" --data_path "$BATCH_ABS" --out_dir "$RUN"
  echo "run $k took $(( $(date +%s) - start )) s"
done
cd "$ROOT"
echo "done: $(ls "$OUT_ABS" | grep -c '^run_') run folders in $OUT"
