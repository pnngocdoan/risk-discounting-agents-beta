"""Run the prediction test: no tools, the model is asked the probability of harm.

  python scripts/60_run_predictions.py --model <served model name> --base-url http://localhost:8000/v1 \
      --in data/conditions/batch_5_1/predict --out results/pilot/batch_5_1/qwen2.5-3b/predict

Every file in --in is one condition (a list of {"id", "prompt"}; scripts/50_build_conditions.py writes
them). The prompt is sent as the one user message, temperature 0, no tools. The answer is read with a
regex ("Probability: x%"); a failed parse is retried up to twice. Rows are appended to
<out>/<condition>.jsonl by uid, so a rerun skips finished rows. Prints, per condition, how many
answers parsed and each task's predicted p, and the overall parse rate (the plan wants at least 95%).
"""
import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROB = re.compile(r"probability\s*:?\s*\**\s*(\d+(?:\.\d+)?)\s*%", re.I)


def ask(base, key, model, prompt, timeout=300):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 300}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]["content"]


def parse(text):
    m = PROB.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 <= v <= 100 else None


def run_one(a, condition, row):
    raw, value = "", None
    for attempt in range(1, 4):
        try:
            raw = ask(a.base_url, a.api_key, a.model, row["prompt"])
        except Exception as e:
            raw = f"ERROR {e}"
        value = parse(raw)
        if value is not None:
            break
    return dict(uid=f"{condition}:{row['id']}", condition=condition, task=row["id"], predicted_p=value,
                parsed=value is not None, attempts=attempt, raw=raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the name the server serves the model under")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default=os.environ.get("RDA_BASE_URL"))
    ap.add_argument("--api-key", default=os.environ.get("RDA_API_KEY", "none"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not a.base_url:
        raise SystemExit("set --base-url (or RDA_BASE_URL) to the model server")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = parsed = 0
    for f in sorted(Path(a.inp).glob("*.json")):
        cond = f.stem
        target = out / f"{cond}.jsonl"
        done = {json.loads(l)["uid"] for l in target.read_text().splitlines() if l.strip()} if target.exists() else set()
        todo = [r for r in json.loads(f.read_text()) if f"{cond}:{r['id']}" not in done]
        with ThreadPoolExecutor(max_workers=a.workers) as ex, open(target, "a") as fh:
            for res in ex.map(lambda r: run_one(a, cond, r), todo):
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
        rows = [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
        total += len(rows)
        parsed += sum(r["parsed"] for r in rows)
        cells = ", ".join(f"{r['task']}={r['predicted_p'] if r['parsed'] else 'unparsed'}" for r in sorted(rows, key=lambda r: r["task"]))
        print(f"{cond:<12} parsed {sum(r['parsed'] for r in rows)}/{len(rows)} | {cells}")
    print(f"\noverall: {parsed} of {total} answers parsed ({100 * parsed / max(total, 1):.0f}%; the plan wants at least 95%). Rows are in {out}")


if __name__ == "__main__":
    main()
