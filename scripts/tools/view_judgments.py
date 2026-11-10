"""Show ShieldAgent's verdicts (safe / unsafe) in an easy-to-read page.

Reads the *_outputs_results.json written by score/eval_with_shield.py. Never modifies it.
Output: a colored, self-contained HTML page (open it, or show it in a notebook), plus a
one-line summary printed to the terminal. Verdicts the judge failed to give (written to
*_fail_cases.json) are listed if you pass --fail-cases.

Usage:
  python scripts/tools/view_judgments.py results/pilot/batch_5_1/qwen2.5-3b/judge/qwen2.5-3b/qwen2.5-3b_generated_trajectories_outputs_results.json
  python scripts/tools/view_judgments.py <results.json> -o verdicts.html --fail-cases <fail_cases.json>
"""
import argparse
import html
import json
from pathlib import Path

CSS = """
body{font-family:system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 12px;color:#222}
h1{font-size:20px} .sum{margin:8px 0 18px;font-size:15px}
.task{border:1px solid #ddd;border-radius:8px;margin:12px 0;padding:10px 14px}
.tag{display:inline-block;padding:2px 10px;border-radius:10px;font-weight:600;color:#fff;margin-right:8px}
.safe{background:#2e7d32}.unsafe{background:#c62828}.none{background:#757575}
pre{white-space:pre-wrap;background:#f6f6f6;border-radius:6px;padding:8px;margin:6px 0}
.lbl{color:#666;font-size:12px;text-transform:uppercase;margin-top:8px}
"""


def verdict_of(item):
    label = item.get("pred_label")
    if label == 1:
        return "unsafe"
    if label == 0:
        return "safe"
    return "none"


def render(items, fails):
    n_safe = sum(verdict_of(i) == "safe" for i in items)
    n_unsafe = sum(verdict_of(i) == "unsafe" for i in items)
    out = [f"<style>{CSS}</style><h1>ShieldAgent verdicts</h1>",
           f"<div class='sum'>{len(items)} judged: <b>{n_unsafe} unsafe</b>, <b>{n_safe} safe</b>"
           f"{f', {len(fails)} with no usable verdict' if fails else ''}</div>"]
    for item in items + fails:
        v = verdict_of(item)
        label = {"safe": "SAFE", "unsafe": "UNSAFE", "none": "NO VERDICT"}[v]
        out.append(f"<div class='task'><span class='tag {v}'>{label}</span>"
                   f"<b>task {html.escape(str(item.get('id')))}</b>")
        out.append(f"<div class='lbl'>Instruction</div><pre>{html.escape(str(item.get('instruction', '')))}</pre>")
        out.append(f"<div class='lbl'>Judge output</div><pre>{html.escape(str(item.get('shield_output', '')))}</pre></div>")
    return "\n".join(out), n_safe, n_unsafe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("-o", "--out", help="output HTML (default: verdicts.html next to the input)")
    ap.add_argument("--fail-cases", help="the *_fail_cases.json from the same run")
    a = ap.parse_args()
    items = json.loads(Path(a.results).read_text())
    fails = json.loads(Path(a.fail_cases).read_text()) if a.fail_cases else []
    page, n_safe, n_unsafe = render(items, fails)
    out = Path(a.out) if a.out else Path(a.results).with_name("verdicts.html")
    out.write_text(page)
    print(f"{len(items)} judged: {n_unsafe} unsafe, {n_safe} safe; wrote {out}")


if __name__ == "__main__":
    main()
