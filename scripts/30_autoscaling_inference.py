"""Run every task of a batch, several times each, in parallel on all the GPUs on the machine.

Used for the pilot's model runs: the 6 history runs per task now, and later the base and T1-T3 runs.

One command does everything: it finds the GPUs, starts one vLLM model server on each, works out
how many jobs each server can take at once, runs every (task, repeat) job, merges the results,
stops the servers and prints what it measured (time, GPU memory, GPU-hours).

Change the model, tasks or repeats with the flags; nothing is fixed:
  --model     Hugging Face id (served locally, e.g. Qwen/Qwen2.5-3B-Instruct) or "api:<name>"
              for a hosted API that is already running (set RDA_BASE_URL and RDA_API_KEY)
  --batch     any batch file (the tasks come from it)
  --repeats   runs per task (default 6)
How many jobs run at once is decided from the hardware and then adjusted as it runs. The first
batch is what vLLM says fits in each GPU (rounded up). After every batch the script reads what
happened (failures, how full vLLM's conversation space got, whether requests had to wait, GPU
utilization) and sets the next batch: bigger if there was room, the same if the GPU is at its
limit, the last good size if jobs failed. One row per batch goes to autoscale_log.csv.
Use --no-autoscale for a fixed size, --start-per-gpu / --max-per-gpu to set the range.

Layout written under --out:
  run_<k>/generated_trajectories.json     all tasks of repeat k (merged at the end)
  run_<k>/parts/<task id>/                 each job's own output and log.txt
  timings.csv, autoscale_log.csv, hardware_report.json   seconds per job; one row per batch; totals
Finished jobs are skipped on a restart.

Usage (from the repo root, after scripts/apply_bug_fixes.sh and `pip install vllm`):
  python scripts/30_autoscaling_inference.py --model Qwen/Qwen2.5-3B-Instruct \
      --batch data/batches/batch_5_1.json --out results/pilot/batch_5_1/qwen2.5-3b/histories
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "third_party" / "Agent-SafetyBench"


# ---------- hardware ----------
def gpu_count():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return 0


class GpuSampler(threading.Thread):
    """Samples nvidia-smi (and the vLLM gauges, if given) every few seconds.

    Keeps run totals (peak memory, average utilization per GPU) and, between calls to
    new_batch() and batch_summary(), the same numbers for just that batch.
    """

    def __init__(self, metrics_urls=(), every=5.0):
        super().__init__(daemon=True)
        self.every, self.stop_flag, self.metrics_urls = every, threading.Event(), list(metrics_urls)
        self.peak, self.utils = {}, {}
        self.new_batch()

    def new_batch(self):
        self.b_utils, self.b_wait, self.b_cache, self.b_mem = [], 0.0, 0.0, 0

    def batch_summary(self):
        return dict(avg_util_pct=round(sum(self.b_utils) / len(self.b_utils)) if self.b_utils else None,
                    peak_waiting=self.b_wait, peak_cache_usage=round(self.b_cache, 3), peak_mem_mib=self.b_mem)

    def run(self):
        while not self.stop_flag.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, check=True).stdout
                for line in out.strip().splitlines():
                    i, u, m = [int(float(x)) for x in line.split(",")]
                    self.peak[i] = max(self.peak.get(i, 0), m)
                    self.utils.setdefault(i, []).append(u)
                    self.b_utils.append(u)
                    self.b_mem = max(self.b_mem, m)
            except Exception:
                pass
            for url in self.metrics_urls:
                g = vllm_gauges(url)
                if g:
                    self.b_wait = max(self.b_wait, g["waiting"])
                    self.b_cache = max(self.b_cache, g["cache_usage"])
            self.stop_flag.wait(self.every)

    def finish(self):
        self.stop_flag.set()
        self.join(timeout=10)
        return {i: dict(peak_mem_mib=self.peak[i], avg_util_pct=round(sum(self.utils[i]) / len(self.utils[i]))) for i in self.peak}


def vllm_gauges(metrics_url):
    """Read vLLM's own gauges: requests waiting for room, and how full the conversation space is (0 to 1).

    nvidia-smi cannot show this: vLLM reserves most of the GPU memory when it starts, so the
    memory always looks nearly full.
    """
    try:
        text = urllib.request.urlopen(metrics_url, timeout=5).read().decode()
    except Exception:
        return None
    def val(pattern):
        m = re.search(r"^" + pattern + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
        return float(m.group(1)) if m else None
    waiting = val(r"vllm:num_requests_waiting")
    cache = val(r"vllm:(?:gpu_cache_usage_perc|kv_cache_usage_perc)")
    if waiting is None and cache is None:
        return None
    return dict(waiting=waiting or 0.0, cache_usage=cache or 0.0)


# ---------- model servers ----------
def reachable(base, key=""):
    """Any HTTP reply means a server is there; only a rejected key is a problem."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
        urllib.request.urlopen(req, timeout=10).read()
        return None
    except urllib.error.HTTPError as e:
        return f"the server at {base} rejected the API key (HTTP {e.code})" if e.code in (401, 403) else None
    except Exception as e:
        return f"cannot reach the model server at {base}: {e}"


def start_server(hf_id, name, gpu, port, max_len, gpu_mem, log_dir):
    log = log_dir / f"vllm_gpu{gpu}.log"
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", hf_id, "--served-model-name", name,
           "--dtype", "half",  # the T4 has no bf16
           "--max-model-len", str(max_len), "--gpu-memory-utilization", str(gpu_mem),
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--port", str(port)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    return proc, log


def wait_for_server(proc, port, log, minutes=15):
    for _ in range(minutes * 12):
        if proc.poll() is not None:
            sys.exit(f"the model server on port {port} stopped while loading; last lines of {log}:\n" + "".join(log.read_text().splitlines(True)[-15:]))
        if reachable(f"http://127.0.0.1:{port}/v1") is None:
            return
        time.sleep(5)
    sys.exit(f"the model server on port {port} did not come up in {minutes} min; see {log}")


def reported_concurrency(log):
    """vLLM logs e.g. 'Maximum concurrency for 12,288 tokens per request: 4.50x' once it knows its memory budget."""
    m = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", log.read_text())
    return float(m[-1]) if m else None


# ---------- jobs ----------
def run_job(model, k, task, out, greedy, timeout, base_url, api_key):
    part = out / f"run_{k}" / "parts" / str(task["id"])
    result = part / "generated_trajectories.json"
    if result.exists():
        return k, task["id"], 0.0, "skipped"
    part.mkdir(parents=True, exist_ok=True)
    work = out / "_work" / f"run_{k}_task_{task['id']}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # eval.py expects ../environments next to its own folder and writes logs into its working folder.
    shutil.copytree(BENCH / "evaluation", work / "evaluation", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.symlink(BENCH / "environments", work / "environments")
    (work / "task.json").write_text(json.dumps([task]))
    cmd = [sys.executable, "eval.py", "--model_name", model, "--greedy", str(greedy),
           "--data_path", str(work / "task.json"), "--out_dir", str(part.resolve())]
    env = dict(os.environ, RDA_BASE_URL=base_url, RDA_API_KEY=api_key)
    t0 = time.time()
    with open(part / "log.txt", "w") as log:
        try:
            rc = subprocess.run(cmd, cwd=work / "evaluation", env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    secs = time.time() - t0
    shutil.rmtree(work, ignore_errors=True)
    ok = rc == 0 and result.exists()
    return k, task["id"], secs, "ok" if ok else f"FAILED ({'timed out after %d s' % timeout if rc == 'timeout' else 'exit %s' % rc}; see {part / 'log.txt'})"


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
    ap.add_argument("--model", required=True, help="Hugging Face id to serve locally, or api:<name> for a hosted API")
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--start-per-gpu", type=int, default=None, help="jobs per GPU in the first batch (default: what vLLM says fits, rounded up)")
    ap.add_argument("--max-per-gpu", type=int, default=32, help="the batch size never grows past this many jobs per GPU")
    ap.add_argument("--no-autoscale", action="store_true", help="keep every batch the same size as the first")
    ap.add_argument("--cache-limit", type=float, default=0.85, help="hold the size when the conversation space got fuller than this")
    ap.add_argument("--max-len", type=int, default=12288, help="longest conversation the server accepts (tokens)")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="share of each GPU's memory the server may use")
    ap.add_argument("--job-timeout", type=int, default=1800, help="seconds before one job is stopped and counted as failed")
    ap.add_argument("--greedy", type=int, default=0, help="0 = random sampling (default), 1 = greedy")
    ap.add_argument("--keep-servers", action="store_true", help="leave the model servers running at the end")
    a = ap.parse_args()

    tasks = json.loads(Path(a.batch).read_text())
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, t) for k in range(1, a.repeats + 1) for t in tasks]

    procs, servers, report = [], [], {}
    try:
        if a.model.startswith("api:"):
            base, key = os.environ.get("RDA_BASE_URL"), os.environ.get("RDA_API_KEY", "")
            if not base:
                sys.exit("--model api:<name> needs RDA_BASE_URL (and RDA_API_KEY) to be set")
            problem = reachable(base, key)
            if problem:
                sys.exit(f"stopping before any job starts: {problem}")
            eval_model, servers, per_gpu, gpus = a.model, [(base, key)], a.start_per_gpu or 8, 0
        else:
            gpus = gpu_count()
            if not gpus:
                sys.exit("no GPU found (nvidia-smi); use --model api:<name> with a hosted API instead")
            name = a.model.split("/")[-1].lower()
            eval_model = f"api:{name}"
            for g in range(gpus):
                proc, log = start_server(a.model, name, g, 8000 + g, a.max_len, a.gpu_mem, out)
                procs.append((proc, 8000 + g, log))
            print(f"starting {a.model} on {gpus} GPU(s); loading can take several minutes")
            for proc, port, log in procs:
                wait_for_server(proc, port, log)
                servers.append((f"http://127.0.0.1:{port}/v1", "none"))
            conc = [reported_concurrency(log) for _, _, log in procs]
            known = [c for c in conc if c]
            per_gpu = a.start_per_gpu or (max(2, math.ceil(min(known))) if known else 4)
            print(f"each server reports room for {known or 'an unknown number of'} full-length conversations; first batch: {per_gpu} jobs per GPU")
            report["vllm_reported_concurrency"] = conc

        n_srv = len(servers)
        level = per_gpu * n_srv                      # jobs in the current batch
        cap = a.max_per_gpu * n_srv
        can_scale = not a.no_autoscale and bool(procs)   # gauges exist only for servers we started
        metrics = [u.rsplit("/v1", 1)[0] + "/metrics" for u, _ in servers] if procs else []
        sampler = GpuSampler(metrics)
        sampler.start()
        pending = list(jobs)
        print(f"{len(tasks)} tasks x {a.repeats} repeats = {len(jobs)} jobs; first batch {min(level, len(pending))} jobs" + ("" if can_scale else " (fixed size)"))
        t0 = time.time()
        failed, rows, log_rows, good, frozen, batch_no, done = 0, [], [], level, False, 0, 0
        while pending:
            batch, pending = pending[:level], pending[level:]
            batch_no += 1
            sampler.new_batch()
            tb = time.time()
            nfail = 0
            with ThreadPoolExecutor(max_workers=len(batch)) as ex:
                futs = [ex.submit(run_job, eval_model, k, t, out, a.greedy, a.job_timeout, *servers[i % n_srv]) for i, (k, t) in enumerate(batch)]
                for fu in as_completed(futs):
                    k, tid, secs, status = fu.result()
                    rows.append((k, tid, round(secs, 1), status))
                    nfail += status not in ("ok", "skipped")
                    done += 1
                    print(f"[{done}/{len(jobs)}] run {k} task {tid}: {status} ({secs:.0f} s)")
            failed += nfail
            bs = sampler.batch_summary()
            secs_b = time.time() - tb
            # decide the next batch size from what this batch did
            if not can_scale or frozen or not pending:
                decision, nxt = "fixed" if not can_scale else ("kept" if pending else "last batch"), level
            elif nfail:
                decision, nxt, frozen = f"{nfail} job(s) failed: back to {good}, stop growing", good, True
            elif bs["peak_waiting"] > 0 or bs["peak_cache_usage"] > a.cache_limit:
                decision, nxt, frozen = "at capacity (requests waited or conversation space nearly full): hold", level, True
            else:
                good = level
                nxt = min(cap, math.ceil(level * 1.5))
                decision = f"room left: grow to {nxt}" if nxt > level else "at the maximum"
            log_rows.append(dict(batch=batch_no, jobs=len(batch), seconds=round(secs_b), jobs_per_min=round(len(batch) / secs_b * 60, 2),
                                 failed=nfail, next_size=nxt, decision=decision, **bs))
            print(f"batch {batch_no}: {len(batch)} jobs in {secs_b:.0f} s, util {bs['avg_util_pct']}%, space used {bs['peak_cache_usage']}, "
                  f"waiting {bs['peak_waiting']}, failed {nfail} -> {decision}")
            level = nxt
        wall = time.time() - t0
        gpu_stats = sampler.finish()
    finally:
        if not a.keep_servers:
            for proc, _, _ in procs:
                try:
                    os.killpg(proc.pid, 15)
                except Exception:
                    pass

    new = not (out / "timings.csv").exists()
    with open(out / "timings.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "task", "seconds", "status"])
        w.writerows(sorted(rows))
    report.update(model=a.model, gpus=gpus, jobs=len(jobs), final_batch_size=level, batches=batch_no, failed_jobs=failed, wall_minutes=round(wall / 60, 2),
                  jobs_per_minute=round(len(jobs) / wall * 60, 2) if wall else 0, gpu_hours=round(wall / 3600 * gpus, 3), per_gpu=gpu_stats,
                  max_len=a.max_len, gpu_mem=a.gpu_mem)
    report["batches_log"] = log_rows
    (out / "hardware_report.json").write_text(json.dumps(report, indent=1))
    if log_rows:
        with open(out / "autoscale_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    merge(out, tasks, a.repeats)
    print(f"done in {wall / 60:.1f} min wall-clock, {report['gpu_hours']} GPU-hours; {failed} failed job(s)")
    for g, s in gpu_stats.items():
        print(f"  GPU {g}: peak memory {s['peak_mem_mib']} MiB, average utilization {s['avg_util_pct']}%")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
