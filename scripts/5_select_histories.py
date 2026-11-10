"""Split the distinct trajectories into safe and unsafe, using ShieldAgent's verdicts.

  python scripts/5_select_histories.py --runs <unique> --judge <judge>

Run it after ShieldAgent has judged <unique> (the notebook cell with eval_with_shield.py). A run is valid when
it is in <unique> (not a duplicate) and ShieldAgent gave a verdict. Valid runs are copied into <out>/safe and
<out>/unsafe (same layout as unique: run_K/generated_trajectories.json). A task is "ready" with at least
--min-safe safe and --min-unsafe unsafe valid runs (pilot defaults 0 and 2, unsafe histories only; the full
plan uses 10 and 10 within 30 draws). Prints a table and the commands to read one safe and one unsafe run per
task. Writes selection.csv, selection_pool.json, safe/, unsafe/. Never changes the runs.
"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def run_numbers(folder):
    return sorted(int(p.name.split("_")[1]) for p in Path(folder).glob("run_*") if p.is_dir())


def verdicts_for_run(judge_dir, k):
    """{task id: (verdict, one-line explanation)} for one repeat."""
    out = {}
    run_dir = Path(judge_dir) / f"run_{k}"
    for f in sorted(run_dir.rglob("*_outputs_results.json")):
        for it in load(f) or []:
            label = it.get("pred_label")
            v = {1: "unsafe", 0: "safe"}.get(label, "none")
            text = str(it.get("shield_output", ""))
            expl = text.split("[Analysis]", 1)[1].strip().replace("\n", " ")[:200] if "[Analysis]" in text else ""
            out[it.get("id")] = (v, expl)
    for f in sorted(run_dir.rglob("*_fail_cases.json")):
        for it in load(f) or []:
            out.setdefault(it.get("id"), ("none", ""))
    return out


def cmd_select(a):
    runs_dir = Path(a.runs)
    out_dir = Path(a.out) if a.out else runs_dir / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, kept = [], {"safe": {}, "unsafe": {}}
    for k in run_numbers(runs_dir):
        traj = {t["id"]: t for t in (load(runs_dir / f"run_{k}" / "generated_trajectories.json") or [])}
        verdicts = verdicts_for_run(a.judge, k)
        for tid in sorted(traj):
            why, t = [], traj[tid]
            v, expl = verdicts.get(tid, ("none", ""))
            if v == "none":
                why.append("no ShieldAgent verdict")
            rows.append(dict(task=tid, repeat=k, verdict=v, usable=not why, problem="; ".join(why), judge_line=expl))
            if not why:
                kept[v].setdefault(k, []).append(t)
    if not rows:
        raise SystemExit("no trajectories found; did you point --runs at the unique folder?")
    with open(out_dir / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for kind, by_run in kept.items():
        for k, items in by_run.items():
            d = out_dir / kind / f"run_{k}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "generated_trajectories.json").write_text(json.dumps(items, indent=1))
    pool, tasks, ready = {}, sorted({r["task"] for r in rows}), 0
    print(f"{len(tasks)} tasks, {len(run_numbers(runs_dir))} repeats (distinct trajectories only)\n")
    print(f"{'task':>6} | usable | safe | unsafe | problems             | ready for histories?")
    for tid in tasks:
        rs = [r for r in rows if r["task"] == tid]
        safe = [r["repeat"] for r in rs if r["usable"] and r["verdict"] == "safe"]
        unsafe = [r["repeat"] for r in rs if r["usable"] and r["verdict"] == "unsafe"]
        probs = sorted({p for r in rs for p in r["problem"].split("; ") if p})
        ok = len(safe) >= a.min_safe and len(unsafe) >= a.min_unsafe
        ready += ok
        pool[str(tid)] = dict(safe_repeats=safe, unsafe_repeats=unsafe, ready=ok)
        print(f"{tid:>6} | {sum(r['usable'] for r in rs):>6} | {len(safe):>4} | {len(unsafe):>6} | {', '.join(probs)[:20]:<20} | "
              f"{'YES' if ok else 'no: needs ' + str(a.min_safe) + ' safe and ' + str(a.min_unsafe) + ' unsafe'}")
    (out_dir / "selection_pool.json").write_text(json.dumps(pool, indent=1))
    print(f"\n{ready} of {len(tasks)} tasks are ready (at least {a.min_safe} safe and {a.min_unsafe} unsafe usable runs).")
    print("\nSpot-check (read only these): one safe and one unsafe run per task")
    for tid in tasks:
        for kind, reps in (("safe", pool[str(tid)]["safe_repeats"]), ("unsafe", pool[str(tid)]["unsafe_repeats"])):
            if reps:
                print(f"  task {tid}, {kind}, repeat {reps[0]}: python scripts/tools/view_trajectories.py {runs_dir}/run_{reps[0]}/generated_trajectories.json --terminal --ids {tid}")
    print(f"\nwrote {out_dir}/selection.csv, selection_pool.json, and the safe/ and unsafe/ folders (same layout as unique)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the unique folder")
    ap.add_argument("--judge", required=True, help="the ShieldAgent results folder (run_1 ... run_N)")
    ap.add_argument("--min-safe", type=int, default=0)
    ap.add_argument("--min-unsafe", type=int, default=2)
    ap.add_argument("--out", help="default: <runs>/selection")
    cmd_select(ap.parse_args())


if __name__ == "__main__":
    main()
