"""Compare BASE, T1, T2 and T3: which conditions made the agent more (or less) unsafe, and for which task.

  python scripts/70_compare_conditions.py --base results/pilot/batch_5_1/qwen2.5-3b/judge_base/run_1 \
      --conds results/pilot/batch_5_1/qwen2.5-3b/judge_cond

--conds holds one ShieldAgent results folder per condition (T1_p10, T2_k2, T3_k2_p10, ...). Prints one
row per task (U = unsafe, S = safe, - = not run or no verdict), the unsafe count per condition, and the
changes against BASE: safe to unsafe is "more unsafe", unsafe to safe is "less unsafe". Greedy, one run
per task and condition, so this is a count of tasks, not a statistic. Only reads the results.
"""
import argparse
import json
from pathlib import Path


def verdicts(folder):
    """{task id: 'U' or 'S'} from the ShieldAgent results under one folder."""
    out = {}
    for f in sorted(Path(folder).rglob("*_outputs_results.json")):
        for it in json.loads(f.read_text()) or []:
            v = {1: "U", 0: "S"}.get(it.get("pred_label"))
            if v:
                out[it["id"]] = v
    return out


def order(name):
    group = name.split("_")[0]
    nums = [int(x) for x in "".join(c if c.isdigit() else " " for c in name).split()]
    return ({"T1": 1, "T2": 2, "T3": 3}.get(group, 9), nums)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="ShieldAgent results folder for BASE")
    ap.add_argument("--conds", required=True, help="folder with one ShieldAgent results folder per condition")
    a = ap.parse_args()
    cols = {"BASE": verdicts(a.base)}
    for d in sorted((p for p in Path(a.conds).iterdir() if p.is_dir()), key=lambda p: order(p.name)):
        cols[d.name] = verdicts(d)
    tasks = sorted({t for v in cols.values() for t in v})
    names = list(cols)
    w = max(len(n) for n in names)
    print("task | " + " | ".join(n for n in names))
    for t in tasks:
        print(f"{t:>4} | " + " | ".join(cols[n].get(t, "-").center(len(n)) for n in names))
    print("\nunsafe count (tasks judged):")
    for n in names:
        print(f"  {n:<{w}} {sum(v == 'U' for v in cols[n].values())} of {len(cols[n])}")
    print("\nchanges against BASE:")
    base = cols["BASE"]
    for group in ("T1", "T2", "T3"):
        more, less = [], []
        for n in names:
            if not n.startswith(group + "_"):
                continue
            for t, v in cols[n].items():
                if t in base and base[t] == "S" and v == "U":
                    more.append(f"task {t} in {n}")
                if t in base and base[t] == "U" and v == "S":
                    less.append(f"task {t} in {n}")
        print(f"  {group}: more unsafe (S to U): {', '.join(more) or 'none'}")
        print(f"  {group}: less unsafe (U to S): {', '.join(less) or 'none'}")
    could = [t for t in tasks if base.get(t) == "S"]
    print(f"\nTasks that were safe at BASE, so the only ones that can get more unsafe: {could or 'none'}")


if __name__ == "__main__":
    main()
