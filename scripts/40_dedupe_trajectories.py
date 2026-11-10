"""Remove repeated trajectories: if two repeats of a task took exactly the same actions, keep one.

Two trajectories of the same task are "the same" when the agent made the same tool calls with the
same arguments in the same order. The environment's answers are fixed by those calls, so the final
wording of the reply does not make a new history.

Reads the runs from the collection step and writes a copy with only the first occurrence of each
distinct trajectory, in the same layout, so the judges can run on it unchanged. The originals
are never touched.

Usage (from the repo root):
  python scripts/40_dedupe_trajectories.py --runs results/pilot/batch_5_1/qwen2.5-3b/histories
Writes <runs>/../unique/run_1 ... run_N/generated_trajectories.json and <runs>/../unique/dedupe.csv
"""
import argparse
import csv
import json
from pathlib import Path


def actions_of(item):
    sig = []
    for m in item.get("output", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                sig.append((fn.get("name"), str(fn.get("arguments"))))
    return tuple(sig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with run_1 ... run_N")
    ap.add_argument("--out", help="output folder (default: <runs>/../unique)")
    a = ap.parse_args()

    runs = Path(a.runs)
    out = Path(a.out) if a.out else runs.parent / "unique"
    ks = sorted(int(p.name.split("_")[1]) for p in runs.glob("run_*") if p.is_dir())
    seen, rows = {}, []
    for k in ks:
        kept = []
        f = runs / f"run_{k}" / "generated_trajectories.json"
        for item in json.loads(f.read_text()) if f.exists() else []:
            sig = (item["id"], actions_of(item))
            first = seen.setdefault(sig, k)
            keep = first == k
            if keep:
                kept.append(item)
            rows.append(dict(task=item["id"], repeat=k, kept=keep, same_as_repeat="" if keep else first, actions=len(sig[1])))
        (out / f"run_{k}").mkdir(parents=True, exist_ok=True)
        (out / f"run_{k}" / "generated_trajectories.json").write_text(json.dumps(kept, indent=1, ensure_ascii=False))
    with open(out / "dedupe.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["task", "repeat", "kept", "same_as_repeat", "actions"])
        w.writeheader()
        w.writerows(rows)

    print(f"{'task':>6} | repeats | distinct | dropped as the same as another repeat")
    for tid in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == tid]
        d = [f"{r['repeat']}=same as {r['same_as_repeat']}" for r in rs if not r["kept"]]
        print(f"{tid:>6} | {len(rs):>7} | {sum(r['kept'] for r in rs):>8} | {', '.join(d) or '-'}")
    print(f"\nkept {sum(r['kept'] for r in rows)} of {len(rows)} trajectories; wrote {out}")


if __name__ == "__main__":
    main()
