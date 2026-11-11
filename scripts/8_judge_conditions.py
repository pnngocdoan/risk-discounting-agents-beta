"""Judge with ShieldAgent, loading ShieldAgent once. Two uses:

  --histories <folder>   the runs from script 3 (run_1 ... run_N): writes <folder>/../judge_histories/run_k,
                         which script 4 reads as --judge
  --out <folder>         BASE and every condition from script 7 (below)

Both flags can be given in one call.

BASE and conditions:

Reads  <out>/base/run_k and <out>/cond/<name>/run_k  (written by 7_run_conditions.py; every run_k that exists)
Writes <out>/judge_base/run_k and, for run 1, <out>/judge_cond/<name>; for run k >= 2, <out>/judge_cond_run_k/<name>
(the folders 9_compare_conditions.py reads)

Uses the same judge code and settings as the one-file command (explanation of 60 tokens after the verdict).
A folder that already has results is skipped. Run it after the model servers have stopped.
"""
import argparse, datetime, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "third_party" / "Agent-SafetyBench" / "score"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histories", help="folder with run_1 ... run_N from script 3; judged into <folder>/../judge_histories")
    ap.add_argument("--out", help="results folder, e.g. results/pilot/batch_5_1/qwen2.5-3b")
    ap.add_argument("--model-path", default="thu-coai/ShieldAgent")
    ap.add_argument("--target-model-name", help="label in the output file names (default: the last part of --out, e.g. qwen2.5-3b)")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--explain-tokens", type=int, default=60)
    a = ap.parse_args()

    if not a.out and not a.histories:
        sys.exit("give --histories <folder> and/or --out <folder>")
    out = Path(a.out or a.histories).resolve()
    a.target_model_name = a.target_model_name or (out.name if a.out else out.parent.name)
    todo = []   # (name, trajectories folder, judge folder)
    if a.histories:
        h = Path(a.histories).resolve()
        for d in sorted(h.glob("run_*"), key=lambda p: int(p.name.split("_")[1])):
            if (d / "generated_trajectories.json").exists():
                todo.append((d.name, d, h.parent / "judge_histories" / d.name))
    def run_no(p):
        return int(p.name.split("_")[1])
    if a.out:
        for d in sorted((out / "base").glob("run_*"), key=run_no):
            if (d / "generated_trajectories.json").exists():
                todo.append(("BASE" if run_no(d) == 1 else f"BASE run {run_no(d)}", d, out / "judge_base" / d.name))
        for d in sorted((out / "cond").glob("*/run_*"), key=lambda p: (run_no(p), p.parent.name)):
            if (d / "generated_trajectories.json").exists():
                k = run_no(d)
                todo.append((d.parent.name if k == 1 else f"{d.parent.name} run {k}", d,
                             out / ("judge_cond" if k == 1 else f"judge_cond_run_{k}") / d.parent.name))
    todo = [t for t in todo if not glob.glob(f"{t[2]}/**/*_outputs_results.json", recursive=True)]
    if not todo:
        sys.exit("nothing to judge: no trajectory files, or all are already judged")
    print(f"judging {len(todo)}: {', '.join(t[0] for t in todo)}")

    os.chdir(SCORE)
    sys.path.insert(0, str(SCORE))
    import eval_with_shield as ev

    ev.generation_config["max_new_tokens"] = 10
    ev.ANSWER_ORDER = "answer_first"
    ev.EXPLAIN_TOKENS = a.explain_tokens
    ev.REASONING_TOKENS = 120
    model, tokenizer, device = ev.create_model_tokenizer(a.model_path, a.model_path)
    stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    for name, traj, jdir in todo:
        ev.SHIELD_OUT = str(jdir)
        os.makedirs(f"{jdir}/{a.target_model_name}", exist_ok=True)
        print(f"=== JUDGE {name} ===", flush=True)
        ev.run_each_file("generated_trajectories", str(traj), stamp, a.target_model_name, model, tokenizer, device,
                         "qwen", "", a.batch_size, a.target_model_name)
    print("done")


if __name__ == "__main__":
    main()
