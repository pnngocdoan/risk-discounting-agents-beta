"""Run every (task, repeat) pair of a batch at the same time against one model server.

Why: a single eval.py run does one conversation at a time and leaves the GPU mostly idle.
The 5 tasks x 6 repeats are independent, so this starts many eval.py jobs at once; the server
(scripts/30_serve_model.sh, or any OpenAI-style API such as Mistral's) batches the requests.

Each job is one task in one repeat, in its own working folder (eval.py writes log files into its
working folder, so jobs must not share one). Finished jobs are skipped on a restart.
Random sampling is used (--greedy 0), so repeats differ.

Layout written under --out:
  run_<k>/generated_trajectories.json     all tasks of repeat k, in batch order (merged at the end)
  run_<k>/parts/<task id>/                 each job's own output and log.txt
  timings.csv                              seconds per job

Usage (from the repo root, after scripts/apply_bug_fixes.sh and with the server running):
  export RDA_BASE_URL=http://127.0.0.1:8000/v1 RDA_API_KEY=none
  python scripts/31_run_parallel.py --model api:qwen2.5-3b --batch data/batches/batch_5_1.json \
      --out results/pilot/batch_5_1/qwen2.5-3b/histories --repeats 6 --workers 12
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "third_party" / "Agent-SafetyBench"


def run_job(model, k, task, out, greedy, timeout):
    part = out / f"run_{k}" / "parts" / str(task["id"])
    result = part / "generated_trajectories.json"
    if result.exists():
        return k, task["id"], 0.0, "skipped"
    part.mkdir(parents=True, exist_ok=True)
    work = out / "_work" / f"run_{k}_task_{task['id']}"
    if work.exists():
        shutil.rmtree(work)
    (work).mkdir(parents=True)
    # eval.py expects ../environments next to its own folder and writes logs into its working folder.
    shutil.copytree(BENCH / "evaluation", work / "evaluation", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.symlink(BENCH / "environments", work / "environments")
    (work / "task.json").write_text(json.dumps([task]))
    cmd = [sys.executable, "eval.py", "--model_name", model, "--greedy", str(greedy),
           "--data_path", str(work / "task.json"), "--out_dir", str(part.resolve())]
    t0 = time.time()
    with open(part / "log.txt", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=work / "evaluation", stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    secs = time.time() - t0
    shutil.rmtree(work, ignore_errors=True)
    ok = rc == 0 and result.exists()
    return k, task["id"], secs, "ok" if ok else f"FAILED ({'timed out after %d s' % timeout if rc == 'timeout' else 'exit %s' % rc}; see {part / 'log.txt'})"


def server_is_up():
    """Quick test that the OpenAI-style server answers, so a dead server fails fast instead of hanging."""
    base = os.environ.get("RDA_BASE_URL")
    if not base:
        return "RDA_BASE_URL is not set (export RDA_BASE_URL=http://127.0.0.1:8000/v1 RDA_API_KEY=none)"
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + os.environ.get("RDA_API_KEY", "")})
        urllib.request.urlopen(req, timeout=10).read()
        return None
    except urllib.error.HTTPError as e:
        # Any HTTP reply means the server is there; only a rejected key is a problem.
        if e.code in (401, 403):
            return f"the server at {base} rejected the API key (HTTP {e.code})"
        return None
    except Exception as e:
        return f"cannot reach the model server at {base}: {e}"


def merge(out, tasks, repeats):
    for k in range(1, repeats + 1):
        merged, missing = [], []
        for t in tasks:
            f = out / f"run_{k}" / "parts" / str(t["id"]) / "generated_trajectories.json"
            if f.exists():
                merged.extend(json.loads(f.read_text()))
            else:
                missing.append(t["id"])
        (out / f"run_{k}").mkdir(parents=True, exist_ok=True)
        (out / f"run_{k}" / "generated_trajectories.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False))
        print(f"run_{k}: {len(merged)} task(s) merged" + (f", MISSING {missing}" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="eval.py model name, e.g. api:qwen2.5-3b")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12, help="jobs running at the same time")
    ap.add_argument("--job-timeout", type=int, default=1800, help="seconds before one job is stopped and counted as failed")
    ap.add_argument("--greedy", type=int, default=0, help="0 = random sampling (default), 1 = greedy")
    a = ap.parse_args()

    if a.model.startswith("api:"):
        problem = server_is_up()
        if problem:
            sys.exit(f"stopping before any job starts: {problem}")
    tasks = json.loads(Path(a.batch).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, t) for k in range(1, a.repeats + 1) for t in tasks]
    print(f"{len(tasks)} tasks x {a.repeats} repeats = {len(jobs)} jobs, {a.workers} at a time")

    t0 = time.time()
    failed = 0
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_job, a.model, k, t, out, a.greedy, a.job_timeout) for k, t in jobs]
        for n, fu in enumerate(as_completed(futs), 1):
            k, tid, secs, status = fu.result()
            rows.append((k, tid, round(secs, 1), status))
            failed += not status in ("ok", "skipped")
            print(f"[{n}/{len(jobs)}] run {k} task {tid}: {status} ({secs:.0f} s)")
    new = not (out / "timings.csv").exists()
    with open(out / "timings.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "task", "seconds", "status"])
        w.writerows(sorted(rows))

    merge(out, tasks, a.repeats)
    print(f"done in {(time.time() - t0) / 60:.1f} min wall-clock; {failed} failed job(s)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
