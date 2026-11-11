"""Compare BASE, T1, T2 and T3: which conditions made the agent more (or less) unsafe, and for which task.

  python scripts/9_compare_conditions.py --base results/pilot/batch_5_1/qwen2.5-3b/judge_base/run_1 \
      --conds results/pilot/batch_5_1/qwen2.5-3b/judge_cond

--conds holds one ShieldAgent results folder per condition (T1_p10, T2_k2, T3_k2_p10, ...). Prints one
row per task (U = unsafe, S = safe, - = not run or no verdict), the unsafe count per condition, and the
changes against BASE: safe to unsafe is "more unsafe", unsafe to safe is "less unsafe". With one run per task and
condition this is a count of tasks, not a statistic.

With several runs per task (7_run_conditions.py --repeats 3) the verdict of a task is the majority of its runs
(unsafe if more than half of the runs are unsafe); the tab "results" also shows how many runs were unsafe
("2 of 3") for the condition, BASE and T0. The other runs are read from judge_base/run_k and judge_cond_run_k
(written by 8_judge_conditions.py); the folders --base and --conds are run 1.

With --audit <audit.xlsx> it also writes the tab "results": one row per condition and task with the ShieldAgent
verdict (safe / unsafe), the BASE verdict and the change against BASE, the T0 verdict and the change against T0 (more unsafe / less unsafe / same) and an empty
"analysis" column for your notes. Run it again after a new judging: the verdicts are refreshed, the analysis cells
you typed stay.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
import audit_sheet as sheet


def verdicts(folder):
    """{task id: 'U' or 'S'} from the ShieldAgent results under one folder."""
    out = {}
    for f in sorted(Path(folder).rglob("*_outputs_results.json")):
        for it in json.loads(f.read_text()) or []:
            v = {1: "U", 0: "S"}.get(it.get("pred_label"))
            if v:
                out[it["id"]] = v
    return out


def runs_verdicts(base_dir, conds_dir):
    """[(base folder, conds folder) for run 1, 2, ...]: run 1 is what was given, the others sit next to it."""
    base_dir, conds_dir = Path(base_dir), Path(conds_dir)
    pairs = [(base_dir, conds_dir)]
    k = 2
    while (base_dir.parent / f"run_{k}").exists() or (conds_dir.parent / f"judge_cond_run_{k}").exists():
        pairs.append((base_dir.parent / f"run_{k}", conds_dir.parent / f"judge_cond_run_{k}"))
        k += 1
    return pairs


def order(name):
    group = name.split("_")[0]
    nums = [int(x) for x in "".join(c if c.isdigit() else " " for c in name).split()]
    return ({"T0": 0, "T1": 1, "T2": 2, "T3": 3}.get(group, 9), nums)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="ShieldAgent results folder for BASE")
    ap.add_argument("--conds", required=True, help="folder with one ShieldAgent results folder per condition")
    ap.add_argument("--audit", help="audit.xlsx: also write the tab 'results' there")
    a = ap.parse_args()
    per_run = {}   # condition -> [verdicts of run 1, run 2, ...]
    for bdir, cdir in runs_verdicts(a.base, a.conds):
        per_run.setdefault("BASE", []).append(verdicts(bdir))
        if cdir.is_dir():
            for d in sorted((p for p in cdir.iterdir() if p.is_dir()), key=lambda p: order(p.name)):
                per_run.setdefault(d.name, []).append(verdicts(d))
    nruns = len(runs_verdicts(a.base, a.conds))
    unsafe_n, total_n = {}, {}   # (condition, task) -> unsafe runs, judged runs
    cols = {}
    for n in sorted(per_run, key=lambda n: (-1, []) if n == "BASE" else order(n)):
        cols[n] = {}
        for t in {t for r in per_run[n] for t in r}:
            vs = [r[t] for r in per_run[n] if t in r]
            unsafe_n[n, t], total_n[n, t] = vs.count("U"), len(vs)
            cols[n][t] = "U" if 2 * vs.count("U") > len(vs) else "S"
    if nruns > 1:
        print(f"{nruns} runs per task and condition: the verdict is the majority of the judged runs\n")
    tasks = sorted({t for v in cols.values() for t in v})
    names = list(cols)
    w = max(len(n) for n in names)
    print("task | " + " | ".join(n for n in names))
    for t in tasks:
        print(f"{t:>4} | " + " | ".join(cols[n].get(t, "-").center(len(n)) for n in names))
    print("\nunsafe count (tasks judged):")
    for n in names:
        print(f"  {n:<{w}} {sum(v == 'U' for v in cols[n].values())} of {len(cols[n])}")
    base = cols["BASE"]
    t0 = cols.get("T0", {})

    def changes(ref, label, groups):
        print(f"\nchanges against {label}:")
        for group in groups:
            more, less = [], []
            for n in names:
                if n != group and not n.startswith(group + "_"):
                    continue
                for t, v in cols[n].items():
                    if ref.get(t) == "S" and v == "U":
                        more.append(f"task {t} in {n}")
                    if ref.get(t) == "U" and v == "S":
                        less.append(f"task {t} in {n}")
            print(f"  {group}: more unsafe (S to U): {', '.join(more) or 'none'}")
            print(f"  {group}: less unsafe (U to S): {', '.join(less) or 'none'}")

    changes(base, "BASE", ("T0", "T1", "T2", "T3"))
    if t0:
        changes(t0, "T0 (the risk named, no number; the reference for T1, T2 and T3)", ("T1", "T2", "T3"))
    could = [t for t in tasks if base.get(t) == "S"]
    if a.audit:
        old = {(r["condition"], r["task"]): r for r in sheet.read_tabs(a.audit).get("results", [])}
        word = {"U": "unsafe", "S": "safe"}
        rows = []
        for n in names:
            for t in sorted(cols[n], key=sheet.sort_key):
                b, z, v = base.get(t), t0.get(t), cols[n][t]
                frac = lambda c: f"{unsafe_n[c, t]} of {total_n[c, t]}" if (c, t) in total_n else ""
                diff = lambda ref: "" if not ref else ("same" if ref == v else "more unsafe" if v == "U" else "less unsafe")
                rows.append(dict(condition=n, task=str(t), verdict=word[v], unsafe_runs=frac(n),
                                 base_verdict=word.get(b, ""), base_unsafe_runs=frac("BASE"),
                                 change="" if n == "BASE" else diff(b),
                                 t0_verdict=word.get(z, ""), t0_unsafe_runs=frac("T0"), change_vs_t0="" if n in ("BASE", "T0") or not t0 else diff(z),
                                 analysis=old.get((n, str(t)), {}).get("analysis", "")))
        sheet.write_tab(a.audit, "results", rows)
        print(f"\nwrote the tab 'results' ({len(rows)} rows) in {a.audit}")
    print(f"\nTasks that were safe at BASE, so the only ones that can get more unsafe: {could or 'none'}")


if __name__ == "__main__":
    main()
