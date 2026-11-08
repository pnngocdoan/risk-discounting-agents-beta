#!/usr/bin/env bash
# Apply the numbered patches in patches/ to the pinned benchmark submodule.
# Run from the repo root after `git submodule update --init`.
# It first restores the benchmark's evaluation/ folder to the pinned commit, so
# running it twice always gives the same result.
set -euo pipefail
BENCH=third_party/Agent-SafetyBench
git -C "$BENCH" checkout -q -- evaluation
git -C "$BENCH" clean -fdq evaluation
for p in patches/*.patch; do
  git -C "$BENCH" apply "../../$p"
  echo "applied: $p"
done
