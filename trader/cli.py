"""``python -m trader``: run a step for one account or all of them."""

from __future__ import annotations

import argparse
import json

from trader.broker import BROKERS
from trader.config import load_repo_env, trader_account_ids
from trader.runner import run_check

MARKS = {"ok": "✓", "halted": "✗", "closed": "·", "error": "!"}


def _print_check(record: dict) -> None:
    print(f"[{record['account']}] {MARKS.get(record['status'], '?')} {record['status']} — {record['message']}")
    for check in record.get("checks", []):
        state = {True: "✓", False: "✗", None: " "}[check["ok"]]
        print(f"   {state} {check['label']:<18} 엔진 {check['expected']:>14}  실제 {check['actual']:>14}  {check['diff']:>12}  {check['note']}")
    print(f"   기록: {record['path']}")
    if record.get("store_error"):
        print(f"   ! 기록 저장소에 못 써서 로컬에 남김: {record['store_error']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trader", description="동파법 자동매매 (지금은 조회 전용)")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="달력 확인 + 엔진 장부와 증권사 잔고 대조 (주문 없음)")
    target = check.add_mutually_exclusive_group(required=True)
    target.add_argument("--account", help="config/trader/<id>.json 의 id")
    target.add_argument("--all", action="store_true", help="config/trader 의 모든 계좌")
    check.add_argument("--force", action="store_true", help="휴장일에도 대조 (시험용)")
    check.add_argument("--json", action="store_true", help="기록 전체를 JSON으로 출력")
    sub.add_parser("brokers", help="지원하는 증권사")
    args = parser.parse_args(argv)
    load_repo_env()  # broker keys and DONGPA_TRADER_RUNS from .env

    if args.command == "brokers":
        print("\n".join(sorted(BROKERS)))
        return 0

    ids = trader_account_ids() if args.all else [args.account]
    if not ids:
        print("config/trader/*.json 이 없습니다")
        return 1
    worst = 0
    for account_id in ids:
        record = run_check(account_id, force=args.force)
        if args.json:
            print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        else:
            _print_check(record)
        worst = max(worst, {"ok": 0, "closed": 0, "halted": 2, "error": 3}.get(record["status"], 3))
    return worst
