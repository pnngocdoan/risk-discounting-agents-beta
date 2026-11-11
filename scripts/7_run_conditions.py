"""Run BASE and every condition file against one set of model servers, so the model is loaded once.

Starts one vLLM server per GPU, runs all jobs (every task of BASE, T1, T2, T3) together, then stops
the servers. Each job is run by the same function as in 3_run_tasks.py, so the
results are the same files in the same places:

  <out>/base/run_1/generated_trajectories.json
  <out>/cond/<condition name>/run_1/generated_trajectories.json

With --repeats 3 every task of BASE and of each condition is run 3 times (run_1, run_2, run_3). Use --greedy 0
(random sampling) for that: at temperature 0 the 3 runs would be the same.

Jobs that already have a result are skipped, so a second call only runs what is missing.
"""
import argparse, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("autoscaler", HERE / "3_run_tasks.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id, e.g. Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--base", required=True, help="batch file for BASE, e.g. data/batches/batch_5_1.json")
    ap.add_argument("--conds-dir", required=True, help="folder with the condition files (T1_*.json, T2_*.json, T3_*.json)")
    ap.add_argument("--out", required=True, help="results folder, e.g. results/pilot/batch_5_1/qwen2.5-3b")
    ap.add_argument("--greedy", type=int, default=1, help="1 = temperature 0 (default), 0 = random sampling")
    ap.add_argument("--repeats", type=int, default=1, help="runs per task and condition (run_1 ... run_N), default 1")
    ap.add_argument("--max-len", type=int, default=12288)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--job-timeout", type=int, default=1800)
    ap.add_argument("--job-ram-gb", type=float, default=0.8)
    ap.add_argument("--ram-reserve-gb", type=float, default=3.0)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    work = {"BASE": (Path(a.base), out / "base")}
    for f in sorted(Path(a.conds_dir).glob("T*.json")):
        work[f.stem] = (f, out / "cond" / f.stem)
    jobs = []   # (name, run number, out folder, task)
    for k in range(1, a.repeats + 1):
        for name, (f, folder) in work.items():
            for t in json.loads(f.read_text()):
                jobs.append((name, k, folder, t))
    print(f"{len(work)} conditions, {len(jobs)} jobs in total")

    gpus = ac.gpu_count()
    if not gpus:
        sys.exit("no GPU found (nvidia-smi)")
    name = a.model.split("/")[-1].lower()
    log_dir = out / "_servers"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs, servers = [], []
    t0 = time.time()
    try:
        for g in range(gpus):
            procs.append(ac.start_server(a.model, name, g, 8000 + g, a.max_len, a.gpu_mem, log_dir) + (8000 + g,))
        print(f"loading {a.model} once on {gpus} GPU(s); logs in {log_dir}")
        for proc, log, port in procs:
            ac.wait_for_server(proc, port, log)
            servers.append((f"http://127.0.0.1:{port}/v1", "none"))
        load_s = time.time() - t0

        ram_cap = ac.jobs_that_fit_in_ram(ac.ram_available_gb(), a.job_ram_gb, a.ram_reserve_gb)
        workers = max(1, min(len(jobs), ram_cap, (os.cpu_count() or 1) * 8))
        print(f"running {len(jobs)} jobs, {workers} at a time (limited by RAM and CPU cores)")
        t1 = time.time()
        failed, done = [], 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(ac.run_job, f"api:{name}", 1, t, folder, a.greedy, a.job_timeout, *servers[i % len(servers)]): (cname, t["id"])
                    for i, (cname, folder, t) in enumerate(jobs)}
            for fu in as_completed(futs):
                cname, tid = futs[fu]
                _, _, secs, status = fu.result()
                done += 1
                if status not in ("ok", "skipped"):
                    failed.append((cname, tid, status))
                print(f"[{done}/{len(jobs)}] {cname} task {tid}: {status} ({secs:.0f} s)")
        run_s = time.time() - t1
    finally:
        for proc, _, _ in procs:
            try:
                os.killpg(proc.pid, 15)
            except Exception:
                pass

    for cname, (f, folder) in work.items():
        print(f"{cname}: ", end="")
        ac.merge(folder, json.loads(f.read_text()), a.repeats)
    print(f"done: model load {load_s:.0f} s, {len(jobs)} jobs {run_s:.0f} s, {len(failed)} failed")
    for x in failed:
        print("FAILED:", *x)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
