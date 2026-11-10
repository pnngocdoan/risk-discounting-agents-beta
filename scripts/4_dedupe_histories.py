"""Keep one of each set of identical trajectories.

  python scripts/4_dedupe_histories.py --runs <histories>

Two repeats of a task are the same trajectory when the agent made the same tool calls with the same
arguments in the same order AND ended with the same final reply. Keeps the first and writes a copy with
only distinct trajectories to <histories>/../unique (same layout, so the judges run on it unchanged).
Originals are never touched. Also writes dedupe.csv.
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


def actions_of(item):
    """The agent's tool calls in order (name and arguments): what makes two trajectories 'the same'."""
    sig = []
    for m in item.get("output", []):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                sig.append((fn.get("name"), str(fn.get("arguments"))))
    return tuple(sig)


def final_reply(item):
    """The agent's last message if it ends with a reply instead of a tool call, in lower case with spaces collapsed.

    How a trajectory ends (go ahead, stop, refuse) is part of what the agent did, so it counts too.
    """
    msgs = [m for m in item.get("output", []) if m.get("role") in ("assistant", "tool")]
    if msgs and msgs[-1].get("role") == "assistant" and not msgs[-1].get("tool_calls"):
        return " ".join(str(msgs[-1].get("content") or "").lower().split())
    return ""


def run_numbers(folder):
    return sorted(int(p.name.split("_")[1]) for p in Path(folder).glob("run_*") if p.is_dir())


def cmd_dedupe(a):
    runs = Path(a.runs)
    out = Path(a.out) if a.out else runs.parent / "unique"
    seen, rows = {}, []
    for k in run_numbers(runs):
        kept = []
        for item in load(runs / f"run_{k}" / "generated_trajectories.json") or []:
            sig = (item["id"], actions_of(item), final_reply(item))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the histories folder (run_1 ... run_N)")
    ap.add_argument("--out", help="default: <runs>/../unique")
    cmd_dedupe(ap.parse_args())


if __name__ == "__main__":
    main()
