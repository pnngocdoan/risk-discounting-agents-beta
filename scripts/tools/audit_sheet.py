"""Shared by scripts 3 to 9: the audit workbook (one tab per step), the trajectory text, and the Anthropic API call.

Every tab is saved to the local audit.xlsx. If the environment variables GOOGLE_SHEET_ID (the id in the sheet's URL)
and GOOGLE_SERVICE_ACCOUNT_JSON (the service account's key file, as text) are set, the same tab is also written to
that Google Sheet after every step, and the tabs are read back from the Google Sheet, so edits made there are used.
"""
import json
import os
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
    "results":    (["condition", "task", "verdict", "unsafe_runs", "base_verdict", "base_unsafe_runs", "change", "t0_verdict", "t0_unsafe_runs", "change_vs_t0", "analysis"], [16, 8, 10, 12, 13, 16, 16, 12, 14, 16, 80]),   # step 9
    "predictions": (["condition", "task", "predicted_p", "answer"], [16, 8, 12, 90]),                                   # step 10
}


def ask(url, key, model, effort, prompt, max_tokens=2000, timeout=120):
    """One question to a model. The Anthropic API by default; any other --api-url is read as an OpenAI-style
    chat endpoint (Gemini, Groq, ...), which takes the key as a Bearer token and has no effort setting."""
    # The model may think before it answers, and that thinking counts toward max_tokens, so leave room.
    if "anthropic.com" not in url:
        body = json.dumps({"model": model, "max_tokens": max(max_tokens, 4000), "messages": [{"role": "user", "content": prompt}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json", "Authorization": "Bearer " + key})
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]["message"]["content"].strip()
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


def _google():
    """The Google spreadsheet (gspread) when GOOGLE_SHEET_ID and GOOGLE_SERVICE_ACCOUNT_JSON are set, else None."""
    sid, key = os.environ.get("GOOGLE_SHEET_ID"), os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not (sid and key):
        return None
    try:
        import gspread
        return gspread.service_account_from_dict(json.loads(key)).open_by_key(sid)
    except Exception as e:
        print(f"WARNING: cannot open the Google Sheet ({type(e).__name__}: {e}); using the local audit.xlsx only")
        return None


def read_tabs(path):
    """{tab name: [row dicts, every cell as text]} for the tabs that have rows. Read from the Google Sheet when it is
    set up (so edits made there count), otherwise from the local file; {} if there is neither."""
    out = {}
    sh = _google()
    if sh is not None:
        try:
            for ws in sh.worksheets():
                if ws.title in TABS:
                    vals = ws.get_all_values()
                    if len(vals) > 1:
                        head = vals[0]
                        out[ws.title] = [{c: (r[head.index(c)] if c in head and head.index(c) < len(r) else "") for c in TABS[ws.title][0]}
                                         for r in vals[1:] if r and r[0] != ""]
            return out
        except Exception as e:
            print(f"WARNING: cannot read the Google Sheet ({type(e).__name__}: {e}); reading the local audit.xlsx")
            out = {}
    if not Path(path).exists():
        return out
    wb = load_workbook(path)
    for name, (cols, _) in TABS.items():
        if name in wb.sheetnames:
            rows = list(wb[name].iter_rows(values_only=True))
            head = [str(h) for h in (rows[0] if rows else [])]
            # columns are found by their header name, so a tab written with older columns still reads
            out[name] = [{c: ("" if c not in head or head.index(c) >= len(r) or r[head.index(c)] is None else str(r[head.index(c)])) for c in cols}
                         for r in rows[1:] if r and r[0] is not None]
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
    push_tab(name, tabs[name])


def push_tab(name, rows):
    """Write one tab to the Google Sheet, if it is set up. A failure only prints a warning: the local file is saved."""
    sh = _google()
    if sh is None:
        return
    cols = TABS[name][0]
    try:
        try:
            ws = sh.worksheet(name)
        except Exception:
            ws = sh.add_worksheet(title=name, rows=max(len(rows) + 1, 2), cols=len(cols))
        ws.clear()
        ws.resize(rows=max(len(rows) + 1, 2), cols=len(cols))
        data = [cols] + [[str(r.get(c, ""))[:32000] for c in cols] for r in rows]
        for i in range(0, len(data), 100):   # in pieces, so one request stays small
            ws.update(values=data[i:i + 100], range_name=f"A{i + 1}", value_input_option="RAW")
        ws.freeze(rows=1)
        ws.format("1:1", {"textFormat": {"bold": True}})
        print(f"  Google Sheet: wrote the tab '{name}' ({len(rows)} rows)")
    except Exception as e:
        print(f"WARNING: could not write the tab '{name}' to the Google Sheet ({type(e).__name__}: {e}); it is saved in the local audit.xlsx")


def sort_key(task, run="0"):
    task, run = str(task), str(run)
    return (0, int(task), int(run)) if task.isdigit() and run.isdigit() else (1, task, run)
