"""Judge BASE and every condition with ShieldAgent, loading ShieldAgent once.

Reads  <out>/base/run_1 and <out>/cond/<name>/run_1  (written by 7_run_conditions.py)
Writes <out>/judge_base/run_1 and <out>/judge_cond/<name>  (the folders 9_compare_conditions.py reads)

Uses the same judge code and settings as the one-file command (explanation of 60 tokens after the verdict).
A folder that already has results is skipped. Run it after the model servers have stopped.
"""
import argparse, datetime, glob, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = ROOT / "third_party" / "Agent-SafetyBench" / "score"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="results folder, e.g. results/pilot/batch_5_1/qwen2.5-3b")
    ap.add_argument("--model-path", default="thu-coai/ShieldAgent")
    ap.add_argument("--target-model-name", help="label in the output file names (default: the last part of --out, e.g. qwen2.5-3b)")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--explain-tokens", type=int, default=60)
    a = ap.parse_args()

    out = Path(a.out).resolve()
    a.target_model_name = a.target_model_name or out.name
    todo = []   # (name, trajectories folder, judge folder)
    if (out / "base" / "run_1" / "generated_trajectories.json").exists():
        todo.append(("BASE", out / "base" / "run_1", out / "judge_base" / "run_1"))
    for d in sorted((out / "cond").glob("*/run_1")):
        if (d / "generated_trajectories.json").exists():
            todo.append((d.parent.name, d, out / "judge_cond" / d.parent.name))
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
