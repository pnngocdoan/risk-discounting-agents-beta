"""Shuffle the filtered tasks once and split them into non-overlapping batches.

Reads data/released_data_filtered.json (never modified). After one seeded
shuffle, the first 120 tasks become 12 batches of 10 (batch_10_1 .. batch_10_12,
for testing and debugging) and the next 500 become 10 batches of 50
(batch_50_1 .. batch_50_10). The last 5 tasks go in batch_5_1, so all 625 tasks
are used. No task appears in more than one batch.

Usage: python scripts/10_prepare_batches.py   (run from the repo root)
"""
import json
import random
from pathlib import Path

SEED = 20261106
PLAN = [(10, 12), (50, 10)]  # (batch size, number of batches), cut in this order

src = Path("data/released_data_filtered.json")
out_dir = Path("data/batches")
out_dir.mkdir(parents=True, exist_ok=True)

tasks = json.load(open(src))
random.Random(SEED).shuffle(tasks)

pos = 0
for size, n_batches in PLAN:
    for x in range(1, n_batches + 1):
        batch = tasks[pos : pos + size]
        pos += size
        json.dump(batch, open(out_dir / f"batch_{size}_{x}.json", "w"), ensure_ascii=False, indent=2)
    print(f"{n_batches} batches of {size}")

rest = tasks[pos:]
if rest:
    json.dump(rest, open(out_dir / f"batch_{len(rest)}_1.json", "w"), ensure_ascii=False, indent=2)
    print(f"1 batch of {len(rest)} (the remaining tasks)")
    pos += len(rest)
print(f"used {pos} of {len(tasks)} tasks")
print("batch_10_1 task ids:", [t["id"] for t in json.load(open(out_dir / "batch_10_1.json"))])
