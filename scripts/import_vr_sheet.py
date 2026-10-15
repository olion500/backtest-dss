"""Copy a VR Google Sheet's cycle history into config/vr/<account>.json ("history").

The web VR tab simulates from the account's start_date; the cycles before it come from
the sheet as recorded (V, band, holdings, Pool), so the chart and table go back to the
VR start. Only cycles that start before start_date are copied. The sheet must be
viewable by link.

usage: python scripts/import_vr_sheet.py SHEET_ID [--tab 적립식] [--account main]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engines.config_store import VR_DIR  # noqa: E402

# Cycle rows start at sheet row 7; columns as in docs/vr_sheet.md
COLUMNS = {"week": "E", "start": "F", "end": "G", "v": "H", "v_min": "I", "v_max": "J", "stock_value": "M",
           "shares": "N", "close": "O", "principal": "S", "traded_qty": "T", "traded_amount": "U",
           "deposit": "X", "pool": "Y"}
FIRST_ROW = 7


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def _number(text: str) -> float | None:
    text = text.strip().replace(",", "")
    return float(text) if text else None


def _date(text: str) -> date:
    return datetime.strptime(text.strip().rstrip("."), "%y.%m.%d").date()


def parse_history(rows: list[list[str]], before: date) -> list[dict]:
    history = []
    for row in rows[FIRST_ROW - 1:]:
        cell = {key: row[_col(c)] if _col(c) < len(row) else "" for key, c in COLUMNS.items()}
        if not cell["start"].strip():
            continue
        start = _date(cell["start"])
        if start >= before:
            continue
        item = {"week": int(_number(cell["week"])), "start": start.isoformat(), "end": _date(cell["end"]).isoformat()}
        for key in ("v", "v_min", "v_max", "stock_value", "close", "principal", "traded_amount", "deposit", "pool"):
            item[key] = _number(cell[key]) or 0.0
        item["shares"] = int(_number(cell["shares"]) or 0)
        item["traded_qty"] = int(_number(cell["traded_qty"]) or 0)
        history.append(item)
    return history


def fetch_rows(sheet_id: str, tab: str) -> list[list[str]]:
    url = (f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet="
           f"{urllib.parse.quote(tab)}")
    with urllib.request.urlopen(url, timeout=30) as response:
        return list(csv.reader(io.StringIO(response.read().decode("utf-8"))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sheet_id")
    parser.add_argument("--tab", default="적립식")
    parser.add_argument("--account", default="main")
    args = parser.parse_args()
    path = VR_DIR / f"{args.account}.json"
    account = json.loads(path.read_text(encoding="utf-8"))
    history = parse_history(fetch_rows(args.sheet_id, args.tab), date.fromisoformat(account["start_date"]))
    account["history"] = history
    path.write_text(json.dumps(account, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: {len(history)} cycles {history[0]['start']} ~ {history[-1]['end']}" if history else f"{path}: no cycles")


if __name__ == "__main__":
    main()
