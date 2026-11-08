"""Filter Agent-SafetyBench's released_data.json into released_data_filtered.json.

Reads the original and never modifies it. Writes one new file next to it:
  released_data_filtered.json  the 625 kept tasks, original order and fields unchanged

Usage: python -I apply_metadata_decisions.py <path/to/data_dir>
"""
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
src = data_dir / "released_data.json"
tasks = json.load(open(src))


def tool_using(r):
    return any(e.get("tools") for e in (r.get("environments") or []))


# (reason, keep-if-true) in the order they are applied
rules = [
    ("not completable (fulfillable != 1)", lambda r: r["fulfillable"] == 1),
    ("no tools in environments", tool_using),
    ("pre-written dialog", lambda r: not r.get("dialog")),
    ("only failure mode is 1", lambda r: r["failure_modes"].strip() != "1"),
]

counts = [("start", len(tasks))]
remaining = tasks
for reason, ok in rules:
    remaining = [r for r in remaining if ok(r)]
    counts.append((f"after rule: {reason}", len(remaining)))
kept = remaining

json.dump(kept, open(data_dir / "released_data_filtered.json", "w"), ensure_ascii=False, indent=2)

for name, n in counts:
    print(f"{n:5d}  {name}")
print(f"kept {len(kept)} of {len(tasks)}")
