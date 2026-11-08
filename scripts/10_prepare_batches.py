"""Shuffle the filtered tasks once and cut them into batch files.

Reads data/released_data_filtered.json (never modified) and writes complete
batches of 10, 50 and 100 tasks to data/batches/batch_<size>_<X>.json, where X
counts up from 1. Every size is cut from the top of the same shuffled list, so
batch_10_1 sits inside batch_50_1, which sits inside batch_100_1. A last chunk
smaller than the batch size is not written.

Usage: python scripts/10_prepare_batches.py   (run from the repo root)
"""
import json
import random
from pathlib import Path

SEED = 20261106
SIZES = (10, 50, 100)

src = Path("data/released_data_filtered.json")
out_dir = Path("data/batches")
out_dir.mkdir(parents=True, exist_ok=True)

tasks = json.load(open(src))
random.Random(SEED).shuffle(tasks)

for size in SIZES:
    n_batches = len(tasks) // size
    for x in range(1, n_batches + 1):
        batch = tasks[(x - 1) * size : x * size]
        json.dump(batch, open(out_dir / f"batch_{size}_{x}.json", "w"), ensure_ascii=False, indent=2)
    print(f"size {size:3d}: {n_batches} batches, {len(tasks) - n_batches * size} tasks left over")

print("batch_10_1 task ids:", [t["id"] for t in tasks[:10]])
