"""Shared by scripts 5 and 6: the audit spreadsheet (read, write) and the Anthropic API call."""
import json
import re
import urllib.request

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

TASK_COLS = ["task", "instruction", "risks", "risky_action", "harm", "p_line"]
TRAJ_COLS = ["task", "run", "verdict", "completed", "validity_reason", "outcome_line", "use", "history_text"]
WIDTHS = {"tasks": [8, 60, 30, 50, 30, 70], "trajectories": [8, 6, 9, 10, 50, 50, 6, 90]}


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


def read_audit(path):
    """({task: row}, {(task, run): row}) from the sheets, empty if the file or a sheet is missing. Every cell is text."""
    tasks, trajs = {}, {}
    try:
        wb = load_workbook(path)
    except Exception:
        return tasks, trajs
    for name, cols, store in (("tasks", TASK_COLS, tasks), ("trajectories", TRAJ_COLS, trajs)):
        if name not in wb.sheetnames:
            continue
        for r in wb[name].iter_rows(min_row=2, values_only=True):
            if r and r[0] is not None:
                d = {c: ("" if v is None else str(v)) for c, v in zip(cols, r)}
                store[d["task"] if name == "tasks" else (d["task"], d["run"])] = d
    return tasks, trajs


def write_audit(path, tasks, trajs):
    wb = Workbook()
    wb.remove(wb.active)
    for name, cols, rows in (("tasks", TASK_COLS, list(tasks.values())), ("trajectories", TRAJ_COLS, list(trajs.values()))):
        ws = wb.create_sheet(name)
        ws.append(cols)
        for r in rows:
            ws.append([r.get(c, "") for c in cols])
        for c in ws[1]:
            c.font = Font(bold=True)
        for col, w in zip("ABCDEFGH", WIDTHS[name]):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def sort_key(k):
    task, run = k
    return (0, int(task), int(run)) if task.isdigit() and run.isdigit() else (1, task, run)
