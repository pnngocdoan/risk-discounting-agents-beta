"""Turn a generated_trajectories.json (trajectories from eval.py) into something easy to read.

Default: a colored, self-contained HTML page (open it in a browser, or show it in a notebook).
--terminal: print the same thing with colors in a terminal or notebook cell.
Reads the file only; never modifies it.

Usage:
  python scripts/view_trajectories.py results/pilot/batch_5_1/qwen2.5-3b/generated_trajectories.json
  python scripts/view_trajectories.py generated_trajectories.json -o view.html
  python scripts/view_trajectories.py generated_trajectories.json --terminal --max-chars 800
  python scripts/view_trajectories.py generated_trajectories.json --ids 12 877
"""
import argparse
import html
import json
import sys
from pathlib import Path


def pretty(x):
    """Pretty-print JSON, including JSON that arrives as a string."""
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return x
    return json.dumps(x, indent=2, ensure_ascii=False)


def parse_messages(task):
    """Turn one task's output messages into simple (kind, title, text) blocks, plus flags."""
    blocks, flags = [], []
    n_assistant = n_calls = 0
    msgs = task.get("output", [])
    for m in msgs:
        role = m.get("role")
        if role == "system":
            blocks.append(("system", "System prompt", m.get("content", "")))
        elif role == "user":
            blocks.append(("user", "User", m.get("content", "")))
        elif role == "assistant" and m.get("tool_calls"):
            n_assistant += 1
            for tc in m["tool_calls"]:
                n_calls += 1
                fn = tc.get("function", {})
                blocks.append(("call", f"Tool call: {fn.get('name')}", pretty(fn.get("arguments", {}))))
        elif role == "assistant":
            n_assistant += 1
            text = m.get("content") or ""
            if "<tool_call>" in text or '{"name"' in text:
                flags.append("tool-call text in final answer")
                blocks.append(("bad", "Assistant (final answer contains tool-call text)", text))
            else:
                blocks.append(("assistant", "Assistant (final answer)", text))
        elif role == "tool":
            blocks.append(("result", f"Tool result: {m.get('name')}", pretty(m.get("content", ""))))
        else:
            blocks.append(("system", f"{role}", str(m)))
    if msgs and (msgs[-1].get("role") != "assistant" or not (msgs[-1].get("content") or "")):
        flags.append("no final answer")
    if n_assistant >= 10:
        flags.append("hit the 10-round limit")
    return blocks, flags, n_assistant, n_calls


def tools_of(task):
    out = []
    for e in task.get("environments") or []:
        out += e.get("tools") or []
    return out


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f4f5f7;color:#1f2328}
header{background:#1f2328;color:#fff;padding:16px 28px}
header h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:#b6bcc4;font-size:13px}
main{max-width:980px;margin:0 auto;padding:20px}
table{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #e6e8eb} th{background:#eef0f3}
a{color:#0b57d0;text-decoration:none}
.task{background:#fff;border-radius:10px;margin:22px 0;padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.task h2{margin:0 0 6px;font-size:18px}
.meta{font-size:13px;color:#57606a;margin:2px 0}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;margin-right:6px;background:#e6e8eb}
.flag{background:#ffd8d3;color:#8c1d13}
.safe{background:#cdeed6;color:#14532d} .unsafe{background:#ffd8d3;color:#8c1d13}
.msg{border-left:5px solid;border-radius:6px;margin:10px 0;padding:8px 12px}
.msg b{display:block;font-size:12px;letter-spacing:.04em;text-transform:uppercase;margin-bottom:4px}
.msg pre{margin:0;white-space:pre-wrap;word-break:break-word;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px}
.user{background:#e8f0fe;border-color:#1a73e8} .user b{color:#1a73e8}
.assistant{background:#e6f4ea;border-color:#188038} .assistant b{color:#188038}
.call{background:#fef3e0;border-color:#e37400} .call b{color:#b45f00}
.result{background:#f3e8fd;border-color:#8430ce} .result b{color:#8430ce}
.bad{background:#fde7e9;border-color:#d93025} .bad b{color:#d93025}
.system{background:#eceff1;border-color:#78909c} .system b{color:#546e7a}
details summary{cursor:pointer;font-size:13px;color:#57606a;margin:6px 0}
"""


def render_html(tasks, source):
    e = html.escape
    rows, body = [], []
    for t in tasks:
        blocks, flags, n_assist, n_calls = parse_messages(t)
        tid = t.get("id")
        label = t.get("label")
        badge = ""
        if label in (0, 1):
            badge = f'<span class="tag {"unsafe" if label == 1 else "safe"}">{"unsafe" if label == 1 else "safe"}</span>'
        flag_html = "".join(f'<span class="tag flag">{e(f)}</span>' for f in flags)
        rows.append(
            f'<tr><td><a href="#task-{tid}">Task {tid}</a></td><td>{n_assist}</td><td>{n_calls}</td>'
            f'<td>{(badge + flag_html) or "-"}</td></tr>'
        )
        parts = [f'<section class="task" id="task-{tid}"><h2>Task {tid} {badge}{flag_html}</h2>']
        parts.append(f'<div class="meta">Risk: {e(", ".join(t.get("risks", [])))} &middot; '
                     f'Failure modes: {e(str(t.get("failure_modes")))} &middot; '
                     f'Tools: {e(", ".join(tools_of(t)))}</div>')
        parts.append(f'<div class="meta">{n_assist} model replies, {n_calls} tool calls</div>')
        for kind, title, text in blocks:
            pre = f'<pre>{e(text)}</pre>'
            if kind == "system":
                parts.append(f'<details><summary>{e(title)} (click to open)</summary>'
                             f'<div class="msg system">{pre}</div></details>')
            else:
                parts.append(f'<div class="msg {kind}"><b>{e(title)}</b>{pre}</div>')
        parts.append("</section>")
        body.append("".join(parts))
    n_flagged = sum(1 for r in rows if "flag" in r)
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><title>Trajectories</title><style>{CSS}</style></head><body>'
        f'<header><h1>Trajectories: {len(tasks)} tasks</h1><p>{e(str(source))} &middot; {n_flagged} with flags</p></header>'
        f'<main><table><tr><th>Task</th><th>Model replies</th><th>Tool calls</th><th>Notes</th></tr>{"".join(rows)}</table>'
        f'{"".join(body)}</main></body></html>'
    )


ANSI = {"user": "34", "assistant": "32", "call": "33", "result": "35", "bad": "31", "system": "90"}


def render_terminal(tasks, max_chars):
    out = []
    for t in tasks:
        blocks, flags, n_assist, n_calls = parse_messages(t)
        out.append(f'\n\033[1m=== Task {t.get("id")}  ({n_assist} replies, {n_calls} tool calls) ===\033[0m')
        if flags:
            out.append("\033[31;1mFLAGS: " + "; ".join(flags) + "\033[0m")
        for kind, title, text in blocks:
            if kind == "system":
                continue  # long and the same for every task; use the HTML view to read it
            text = text if len(text) <= max_chars else text[:max_chars] + f"... [{len(text) - max_chars} more characters]"
            out.append(f'\033[{ANSI[kind]};1m[{title}]\033[0m')
            out.append(f'\033[{ANSI[kind]}m{text}\033[0m')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="generated_trajectories.json from eval.py")
    ap.add_argument("-o", "--out", help="HTML output file (default: next to the input, view.html)")
    ap.add_argument("--terminal", action="store_true", help="print colored text instead of writing HTML")
    ap.add_argument("--max-chars", type=int, default=1500, help="terminal mode: cut each message at this length")
    ap.add_argument("--ids", type=int, nargs="*", help="only these task ids")
    a = ap.parse_args()

    tasks = json.load(open(a.path))
    if a.ids:
        tasks = [t for t in tasks if t.get("id") in set(a.ids)]
    if a.terminal:
        print(render_terminal(tasks, a.max_chars))
        return
    out = Path(a.out) if a.out else Path(a.path).with_name("view.html")
    out.write_text(render_html(tasks, a.path), encoding="utf-8")
    print(f"wrote {out} ({len(tasks)} tasks)")


if __name__ == "__main__":
    sys.exit(main())
