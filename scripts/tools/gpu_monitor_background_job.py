"""GPU monitor: log GPU memory and utilization while a run is going, then summarize it.

Reads nvidia-smi only; never touches your runs. Works from a notebook cell or a terminal.

  start    begin logging in the background (one row per GPU every --interval seconds)
  mark     write a label into the log (e.g. "generate", "judge") so stages can be timed
  stop     stop the background logger
  summary  print per-GPU peak memory, average utilization, temperature, power, and
           GPU-hours; plus the same numbers for each marked stage

Usage:
  python scripts/tools/gpu_monitor_background_job.py start   --out results/pilot/gpu_log.csv [--interval 30]
  python scripts/tools/gpu_monitor_background_job.py mark    --out results/pilot/gpu_log.csv generate
  python scripts/tools/gpu_monitor_background_job.py mark    --out results/pilot/gpu_log.csv judge
  python scripts/tools/gpu_monitor_background_job.py stop    --out results/pilot/gpu_log.csv
  python scripts/tools/gpu_monitor_background_job.py summary --out results/pilot/gpu_log.csv

GPU-hours here is wall-clock time x number of GPUs seen (Kaggle's weekly quota counts wall-clock
GPU time, not how busy the GPU was). "avg util" shows how busy the GPUs actually were.
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

QUERY = "index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
HEADER = ["time", "gpu", "util_pct", "mem_used_mib", "mem_total_mib", "temp_c", "power_w"]


def now():
    return datetime.now(timezone.utc)


def read_smi():
    out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 6:
            rows.append(parts)
    return rows


def pid_file(out):
    return Path(str(out) + ".pid")


def marks_file(out):
    return Path(str(out) + ".marks.csv")


def run_loop(out, interval):
    out = Path(out)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
            f.flush()
        while True:
            t = now().isoformat(timespec="seconds")
            try:
                for r in read_smi():
                    w.writerow([t] + r)
                f.flush()
            except Exception as e:  # keep logging even if one read fails
                print(f"nvidia-smi read failed: {e}", file=sys.stderr)
            time.sleep(interval)


def cmd_start(a):
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pf = pid_file(out)
    if pf.exists():
        try:
            os.kill(int(pf.read_text()), 0)
            print(f"already running (pid {pf.read_text().strip()}), log: {out}")
            return
        except (OSError, ValueError):
            pf.unlink()
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_run", "--out", str(out), "--interval", str(a.interval)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    pf.write_text(str(p.pid))
    print(f"monitoring started (pid {p.pid}), every {a.interval}s, log: {out}")


def cmd_mark(a):
    mf = marks_file(a.out)
    new = not mf.exists()
    with open(mf, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "label"])
        w.writerow([now().isoformat(timespec="seconds"), a.label])
    print(f"marked '{a.label}'")


def cmd_stop(a):
    pf = pid_file(a.out)
    if not pf.exists():
        print("no monitor running for this log")
        return
    try:
        os.kill(int(pf.read_text()), signal.SIGTERM)
        print("monitor stopped")
    except (OSError, ValueError):
        print("monitor was not running")
    pf.unlink()


def load(out):
    with open(out) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = datetime.fromisoformat(r["time"])
        r["util"] = float(r["util_pct"])
        r["mem"] = float(r["mem_used_mib"])
        r["memtot"] = float(r["mem_total_mib"])
        r["temp"] = float(r["temp_c"]) if r["temp_c"] not in ("", "[N/A]") else float("nan")
        r["power"] = float(r["power_w"]) if r["power_w"] not in ("", "[N/A]") else float("nan")
    return rows


def stats(rows, label):
    if not rows:
        print(f"{label}: no samples")
        return
    t0, t1 = min(r["t"] for r in rows), max(r["t"] for r in rows)
    secs = (t1 - t0).total_seconds()
    gpus = sorted({r["gpu"] for r in rows}, key=int)
    print(f"{label}: {t0:%H:%M:%S}-{t1:%H:%M:%S} UTC, {secs / 60:.1f} min, {len(gpus)} GPU(s), "
          f"GPU-hours (wall-clock x GPUs) {secs * len(gpus) / 3600:.2f}")
    for g in gpus:
        rg = [r for r in rows if r["gpu"] == g]
        peak = max(r["mem"] for r in rg)
        tot = rg[0]["memtot"]
        temps = [r["temp"] for r in rg if r["temp"] == r["temp"]]
        pows = [r["power"] for r in rg if r["power"] == r["power"]]
        print(f"  GPU {g}: peak memory {peak:.0f} / {tot:.0f} MiB ({100 * peak / tot:.0f}%), "
              f"avg util {sum(r['util'] for r in rg) / len(rg):.0f}%, "
              f"peak temp {max(temps) if temps else float('nan'):.0f} C, "
              f"avg power {sum(pows) / len(pows) if pows else float('nan'):.0f} W")


def cmd_summary(a):
    rows = load(a.out)
    if not rows:
        print("log is empty")
        return
    stats(rows, "whole log")
    mf = marks_file(a.out)
    if mf.exists():
        with open(mf) as f:
            marks = [(datetime.fromisoformat(r["time"]), r["label"]) for r in csv.DictReader(f)]
        end = max(r["t"] for r in rows)
        for i, (t, label) in enumerate(marks):
            nxt = marks[i + 1][0] if i + 1 < len(marks) else end
            print()
            stats([r for r in rows if t <= r["t"] <= nxt], f"stage '{label}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("start", cmd_start), ("stop", cmd_stop), ("summary", cmd_summary), ("mark", cmd_mark), ("_run", None)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name in ("start", "_run"):
            p.add_argument("--interval", type=int, default=30)
        if name == "mark":
            p.add_argument("label")
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    if a.cmd == "_run":
        run_loop(a.out, a.interval)
    else:
        a.fn(a)


if __name__ == "__main__":
    main()
