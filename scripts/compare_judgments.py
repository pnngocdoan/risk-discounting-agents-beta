"""Compare two ShieldAgent runs on the same tasks (for example answer_first vs reasoning_first).

Reads the *_outputs_results.json of each run, and optionally each run's *_fail_cases.json
(answers with no usable verdict). Prints a table and a summary; never modifies the inputs.

Usage:
  python scripts/compare_judgments.py \
    --a results/.../judge_answer_first/<name>_outputs_results.json --a-name answer_first \
    --b results/.../judge_reasoning_first/<name>_outputs_results.json --b-name reasoning_first \
    [--a-fails <fail_cases.json>] [--b-fails <fail_cases.json>]
"""
import argparse
import json
from pathlib import Path


def load(path):
    if not path or not Path(path).exists():
        return []
    return json.loads(Path(path).read_text())


def verdicts(items):
    return {i["id"]: {1: "unsafe", 0: "safe"}.get(i.get("pred_label"), "none") for i in items}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--a-fails")
    ap.add_argument("--b-fails")
    a = ap.parse_args()

    va = verdicts(load(a.a))
    vb = verdicts(load(a.b))
    for f, v in ((a.a_fails, va), (a.b_fails, vb)):
        for item in load(f):
            v.setdefault(item["id"], "none")

    ids = sorted(set(va) | set(vb), key=str)
    w = max(len(a.a_name), len(a.b_name), 6)
    print(f"{'task':>6}  {a.a_name:<{w}}  {a.b_name:<{w}}  same?")
    agree = 0
    for i in ids:
        x, y = va.get(i, "missing"), vb.get(i, "missing")
        same = x == y and x not in ("none", "missing")
        agree += same
        print(f"{i!s:>6}  {x:<{w}}  {y:<{w}}  {'yes' if same else 'NO'}")
    print(f"\nagree on {agree} of {len(ids)} tasks")
    for name, v in ((a.a_name, va), (a.b_name, vb)):
        n_none = sum(1 for x in v.values() if x == "none")
        print(f"{name}: {sum(x == 'unsafe' for x in v.values())} unsafe, "
              f"{sum(x == 'safe' for x in v.values())} safe, {n_none} with no usable verdict")


if __name__ == "__main__":
    main()
