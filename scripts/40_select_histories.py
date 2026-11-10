"""Pick the usable trajectories (candidate histories) from the repeated runs, so you do not read each one.

For every task and every repeat it combines three things:
  1. did the run finish properly (a saved trajectory, no crash, did not hit the round limit),
  2. did it take at least one action, without getting stuck repeating the same action in a row
     (small models often loop on one tool call; that is not a real history),
  3. what did ShieldAgent say (safe / unsafe / no usable verdict).
It also flags runs whose actions are identical to another repeat of the same task, because
repeating the same trajectory adds nothing to a history.

A run is "usable" when it finished, took an action without looping, and has a safe or unsafe verdict.
Not checked yet: whether the run actually completed the task (a run can be judged safe only because
the model got stuck). The plan's edited judge will add a completion label; until then, read the spot-check runs.
A task is "ready" when it has enough usable runs of each kind (defaults for the small pilot:
at least 2 safe and 2 unsafe; the full plan uses 10 and 10 within 30 draws).

Reads only; never changes the runs or the judgments.

Usage (from the repo root):
  python scripts/40_select_histories.py \
      --runs results/pilot/batch_5_1/qwen2.5-3b/unique \
      --judge results/pilot/batch_5_1/qwen2.5-3b/judge \
      --validity results/pilot/batch_5_1/qwen2.5-3b/unique/validity.json
Writes <judge>/../selection/selection.csv (one row per task and repeat) and
selection_pool.json (task -> which repeats are safe / unsafe, distinct actions only).
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


def nonempty_list(path):
    d = load(path)
    return bool(d) if isinstance(d, list) else False


def actions_of(item):
    """The agent's tool calls in order, as a comparable signature."""
    sig = []
    for m in item.get("output", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                sig.append((fn.get("name"), str(fn.get("arguments"))))
    return tuple(sig)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="folder with run_1 ... run_N (the trajectories)")
    ap.add_argument("--judge", required=True, help="folder with run_1 ... run_N (ShieldAgent results)")
    ap.add_argument("--validity", help="validity.json from 41_judge_validity.py; a run must be usable_as_history to count")
    ap.add_argument("--out", help="output folder (default: <judge>/../selection)")
    ap.add_argument("--min-safe", type=int, default=2)
    ap.add_argument("--min-unsafe", type=int, default=2)
    a = ap.parse_args()

    runs_dir, out_dir = Path(a.runs), Path(a.out) if a.out else Path(a.judge).parent / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(int(p.name.split("_")[1]) for p in runs_dir.glob("run_*") if p.is_dir())

    validity = {}
    if a.validity:
        for r in load(a.validity) or []:
            validity[(r["task"], r["repeat"])] = r
    rows, seen = [], {}
    for k in ks:
        traj = {t["id"]: t for t in (load(runs_dir / f"run_{k}" / "generated_trajectories.json") or [])}
        verdicts = verdicts_for_run(a.judge, k)
        # in the deduplicated folder, repeats that were dropped as duplicates are absent on purpose
        for tid in sorted(set(traj) if (runs_dir / "dedupe.csv").exists() else set(traj) | set(verdicts)):
            part = runs_dir / f"run_{k}" / "parts" / str(tid)
            why = []
            t = traj.get(tid)
            if t is None:
                why.append("no saved trajectory")
            if nonempty_list(part / "error_samples.json"):
                why.append("crashed")
            if nonempty_list(part / "exceed_round_samples.json"):
                why.append("hit the round limit")
            acts = actions_of(t) if t else ()
            if t is not None and not acts:
                why.append("took no action")
            if any(acts[i] == acts[i + 1] for i in range(len(acts) - 1)):
                why.append("looped (same action repeated in a row)")
            v, expl = verdicts.get(tid, ("none", ""))
            if v == "none":
                why.append("no usable verdict")
            if a.validity and t is not None:
                vr = validity.get((tid, k))
                if vr is None:
                    why.append("not checked by the validity judge")
                elif vr["usable_as_history"] is None:
                    why.append("validity judge gave no usable answer")
                elif not vr["usable_as_history"]:
                    why.append("validity judge: " + vr["reason"][:60])
            dup = ""
            if acts and not why:
                if (tid, acts) in seen:
                    dup = f"same actions as repeat {seen[(tid, acts)]}"
                else:
                    seen[(tid, acts)] = k
            rows.append(dict(task=tid, repeat=k, verdict=v, actions=len(acts), usable=not why, duplicate_of=dup,
                             problem="; ".join(why), judge_line=expl))

    with open(out_dir / "selection.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pool, tasks = {}, sorted({r["task"] for r in rows})
    print(f"{len(tasks)} tasks x {len(ks)} repeats\n")
    print(f"{'task':>6} | usable | safe | unsafe | duplicates | problems            | ready for histories?")
    ready = 0
    for tid in tasks:
        rs = [r for r in rows if r["task"] == tid]
        good = [r for r in rs if r["usable"] and not r["duplicate_of"]]
        safe = [r["repeat"] for r in good if r["verdict"] == "safe"]
        unsafe = [r["repeat"] for r in good if r["verdict"] == "unsafe"]
        dups = sum(1 for r in rs if r["duplicate_of"])
        probs = sorted({p for r in rs for p in r["problem"].split("; ") if p})
        ok = len(safe) >= a.min_safe and len(unsafe) >= a.min_unsafe
        ready += ok
        pool[str(tid)] = dict(safe_repeats=safe, unsafe_repeats=unsafe, ready=ok)
        print(f"{tid:>6} | {sum(r['usable'] for r in rs):>6} | {len(safe):>4} | {len(unsafe):>6} | {dups:>10} | {', '.join(probs)[:19]:<19} | "
              f"{'YES' if ok else 'no: needs ' + str(a.min_safe) + ' safe and ' + str(a.min_unsafe) + ' unsafe'}")
    (out_dir / "selection_pool.json").write_text(json.dumps(pool, indent=1))
    print(f"\n{ready} of {len(tasks)} tasks are ready (at least {a.min_safe} safe and {a.min_unsafe} unsafe distinct runs).")
    print("\nSpot-check (read only these): one safe and one unsafe run per task")
    for tid in tasks:
        p = pool[str(tid)]
        for kind, reps in (("safe", p["safe_repeats"]), ("unsafe", p["unsafe_repeats"])):
            if reps:
                print(f"  task {tid}, {kind}, repeat {reps[0]}: python scripts/view_trajectories.py {runs_dir}/run_{reps[0]}/generated_trajectories.json --terminal --ids {tid}")
    print(f"\nwrote {out_dir}/selection.csv and selection_pool.json")


if __name__ == "__main__":
    main()
