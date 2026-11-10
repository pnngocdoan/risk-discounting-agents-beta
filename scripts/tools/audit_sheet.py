"""Shared by scripts 3 to 6: the audit workbook (one tab per step), the trajectory text, and the Anthropic API call."""
import json
import re
import urllib.request
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

# tab name: (columns, column widths). One tab per step.
TABS = {
    "runs":       (["task", "run", "instruction", "history_text"], [8, 6, 50, 90]),                                    # step 3
    "filtered":   (["task", "run", "duplicate_of", "completed", "validity_reason", "verdict", "use"], [8, 6, 12, 10, 60, 9, 6]),   # step 4
    "p_lines":    (["task", "instruction", "risks", "risky_action", "harm", "p_line"], [8, 60, 30, 50, 30, 70]),       # step 5
    "histories":  (["task", "order", "run", "outcome_line", "history_text"], [8, 6, 6, 50, 90]),                       # step 5
    "conditions": (["condition", "task", "added_to_prompt"], [16, 8, 120]),                                            # step 6
    "results":    (["condition", "task", "verdict", "base_verdict", "change", "analysis"], [16, 8, 10, 13, 16, 80]),   # step 9
}


def ask(url, key, model, effort, prompt, max_tokens=2000, timeout=120):
    # The model may think before it answers, and that thinking counts toward max_tokens, so leave room.
    body = json.dumps({"model": model, "max_tokens": max_tokens, "output_config": {"effort": effort},
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
    reply = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return "".join(b.get("text", "") for b in reply["content"] if b.get("type") == "text").strip()


def json_of(text):
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


def try_twice(fn):
    for _ in range(2):
        try:
            r = fn()
            if r:
                return r
        except Exception:
            pass
    return None


def history_text_of(item):
    """The agent's actions and what came back, in order, as it goes into the prompt. The final reply is the last action."""
    lines = []
    for m in item.get("output", []):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                lines.append(f"Action: {fn.get('name')}({fn.get('arguments')})")
        elif role == "tool":
            lines.append(f"Observation: {m.get('content')}")
        elif role == "assistant":
            lines.append(f"Action: {m.get('content')}")
    return "\n".join(lines)


def read_tabs(path):
    """{tab name: [row dicts, every cell as text]} for the tabs that exist; {} if the file does not exist."""
    out = {}
    if not Path(path).exists():
        return out
    wb = load_workbook(path)
    for name, (cols, _) in TABS.items():
        if name in wb.sheetnames:
            out[name] = [{c: ("" if v is None else str(v)) for c, v in zip(cols, r)}
                         for r in wb[name].iter_rows(min_row=2, values_only=True) if r and r[0] is not None]
    return out


def write_tab(path, name, rows):
    """Replace one tab, keep the other tabs as they are."""
    path = Path(path)
    tabs = read_tabs(path)
    tabs[name] = rows
    wb = Workbook()
    wb.remove(wb.active)
    for tab, (cols, widths) in TABS.items():
        if tab not in tabs:
            continue
        ws = wb.create_sheet(tab)
        ws.append(cols)
        for r in tabs[tab]:
            ws.append([str(r.get(c, ""))[:32000] for c in cols])   # a cell holds at most 32767 characters
        for c in ws[1]:
            c.font = Font(bold=True)
        for col, w in zip("ABCDEFGH", widths):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def sort_key(task, run="0"):
    task, run = str(task), str(run)
    return (0, int(task), int(run)) if task.isdigit() and run.isdigit() else (1, task, run)
