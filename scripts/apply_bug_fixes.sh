#!/usr/bin/env bash
# Apply the numbered patches in patches/ to the pinned benchmark submodule.
# Run from the repo root after `git submodule update --init`.
# It first restores the benchmark's evaluation/ and score/ folders to the pinned commit, so
# running it twice always gives the same result.
set -euo pipefail
BENCH=third_party/Agent-SafetyBench
git -C "$BENCH" checkout -q -- evaluation score
git -C "$BENCH" clean -fdq evaluation score
for p in patches/*.patch; do
  git -C "$BENCH" apply "../../$p"
  echo "applied: $p"
done
