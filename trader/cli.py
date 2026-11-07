"""``python -m trader``: run a step for one account or all of them."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal

from trader.broker import BROKERS
from trader.broker.base import OrderRequest
from trader.config import load_repo_env, trader_account_ids
from trader.datasync import run_data
from trader.expected import Expected
from trader.notify import build_embeds, crash_embed, order_text, send
from trader.plan import describe
from trader.runner import run_check, run_orders, run_report, run_stop
from trader.stop import StopPlan
from trader.vr import VRPlan, run_vr_reserve

MARKS = {"ok": "✓", "halted": "✗", "closed": "·", "error": "!", "sent": "✓", "declined": "·", "partial": "!", "mismatch": "✗"}
EXIT = {"ok": 0, "closed": 0, "sent": 0, "declined": 0, "halted": 2, "mismatch": 2, "partial": 3, "error": 3}


def _print_check(record: dict) -> None:
    print(f"[{record['account']}] {MARKS.get(record['status'], '?')} {record['status']} — {record['message']}")
    for check in record.get("checks", []):
        state = {True: "✓", False: "✗", None: " "}[check["ok"]]
        print(f"   {state} {check['label']:<18} 엔진 {check['expected']:>14}  실제 {check['actual']:>14}  {check['diff']:>12}  {check['note']}")
    print(f"   기록: {record['path']}")
    if record.get("store_error"):
        print(f"   ! 기록 저장소에 못 써서 로컬에 남김: {record['store_error']}")


def _ask(plan: list[OrderRequest], expected: Expected) -> bool:
    print(f"\n[{expected.account_id}] {expected.symbol} 주문안 {len(plan)}건 (기준 종가 {expected.as_of}, 모드 {expected.mode})")
    for number, order in enumerate(plan, 1):
        print(f"  {number:>3}. {describe(order)}")
    try:
        answer = input(f"이 {len(plan)}건을 증권사로 보낼까요? 보내려면 y: ")
    except EOFError:
        return False
    return answer.strip().lower() == "y"


def _ask_stop(plan: StopPlan, expected: Expected, last) -> bool:
    print(f"\n[{expected.account_id}] {expected.symbol} 현재가 ${last} — 손절 lot {len(plan.lots)}개")
    for lot in plan.lots:
        print(f"   {lot.buy_date} {lot.qty}주 매수가 ${lot.buy_price} 손절가 ${lot.sl}")
    for order in plan.cancel:
        print(f"   취소: 주문번호 {order.order_id} 매도 LOC {order.qty.normalize():f}주 ${order.price}")
    if plan.sell:
        print(f"   전송: {describe(plan.sell)}")
    for order in plan.resend:
        print(f"   재접수 (퉁치기 매도 중 손절 아닌 몸): {describe(order)}")
    for order in plan.buys:
        print(f"   전송 (하한 밑 매수): {describe(order)}")
    question = "익절 주문을 취소하고 MOC로 손절할까요?" if plan.sell else f"하한 밑 매수 {len(plan.buys)}건을 보낼까요?"
    try:
        answer = input(f"{question} 보내려면 y: ")
    except EOFError:
        return False
    return answer.strip().lower() == "y"


def _ask_vr(plan: VRPlan, todo: list[OrderRequest]) -> bool:
    print(f"\n[VR {plan.vr_id}] {plan.symbol} 사이클 {plan.cycle} ({plan.cycle_start} ~ {plan.cycle_end}, 기준 종가 {plan.close_date})")
    print(f"   V ${plan.v:,.2f}, Pool ${plan.pool:,.2f}, 매수 전부 체결 시 ${plan.buy_cost:,.2f}")
    print(f"   매도 가능 {plan.sellable}주" + (f" — 매도 {plan.sells_cut}주는 이 계좌에 없어 뺐습니다" if plan.sells_cut else ""))
    print(f"   기간잔량 지정가 예약, 장 시작 발주: {plan.first_session} ~ {plan.last_session}")
    for number, order in enumerate(todo, 1):
        print(f"  {number:>3}. {describe(order)}")
    try:
        answer = input(f"이 {len(todo)}건을 예약할까요? 보내려면 y: ")
    except EOFError:
        return False
    return answer.strip().lower() == "y"


def _print_orders(record: dict) -> None:
    print(f"[{record['account']}] {MARKS.get(record['status'], '?')} {record['status']} — {record['message']}")
    for order_id in record.get("cancelled", []):
        print(f"   취소: 주문번호 {order_id}")
    if record.get("collar"):
        for item in record["collar"]["moved"]:
            print(f"   지정가 조정 (현재가 ${record['collar']['last']}의 ±19%): {describe(OrderRequest(**item['request']))} → ${item['price']}")
    for item in record.get("sent", []):
        mark = f"주문번호 {item['order_id']}" if "order_id" in item else f"✗ {item['error']}"
        print(f"   {describe(OrderRequest(**item['request']))}  {mark}")
    print(f"   기록: {record['path']}")
    if record.get("store_error"):
        print(f"   ! 기록 저장소에 못 써서 로컬에 남김: {record['store_error']}")


def _print_report(record: dict) -> None:
    print(f"[{record['account']}] {MARKS.get(record['status'], '?')} {record['status']} — {record['message']}")
    if "close" in record:
        source = "체결가" if record["close_source"] == "fill" else "증권사 현재가, 시간외일 수 있음"
        print(f"   {record.get('symbol')} 종가 ${record['close']} ({source})")
    for fill in record.get("fills", []):
        mark = "✓" if fill["ok"] else "✗"
        actual = "목록에 없음" if fill["actual_qty"] is None else f"{Decimal(str(fill['actual_qty'])).normalize():f}주"
        print(f"   {mark} {order_text(fill['request'])}  예상 {fill['expected_qty']}주 / 체결 {actual}")
    for order in record.get("stray", []):
        print(f"   ✗ 기록에 없는 체결: 주문번호 {order['order_id']} {order['side']} {order['filled_qty']}주")
    for lot in record.get("lots_closed", []):
        print(f"   {lot['kind']} {lot['buy_date']} {lot['qty']}주 ${lot['buy_price']} → {lot['pnl']}")
    for warning in record.get("warnings", []):
        print(f"   ! {warning}")
    if "position_qty" in record:
        print(f"   매수(엔진) {record.get('bought_qty')}주 · 보유 {record['position_qty']}주 · 예수금 ${record['cash']}")
    print(f"   기록: {record['path']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trader", description="동파법 자동매매")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="달력 확인 + 엔진 장부와 증권사 잔고 대조 (주문 없음)")
    target = check.add_mutually_exclusive_group(required=True)
    target.add_argument("--account", help="config/trader/<id>.json 의 id")
    target.add_argument("--all", action="store_true", help="config/trader 의 모든 계좌")
    check.add_argument("--force", action="store_true", help="휴장일에도 대조 (시험용)")
    check.add_argument("--json", action="store_true", help="기록 전체를 JSON으로 출력")
    orders = sub.add_parser("orders", help="대조 후 오늘 주문안을 보여 주고, y로 승인하면 증권사로 전송")
    target = orders.add_mutually_exclusive_group(required=True)
    target.add_argument("--account", help="config/trader/<id>.json 의 id")
    target.add_argument("--all", action="store_true", help="config/trader 의 모든 계좌")
    orders.add_argument("--auto", action="store_true", help="묻지 않음: auto_trade 가 켜진 계좌만 전송 (스케줄 실행용)")
    stop = sub.add_parser("stop", help="마감 15분 전: 손절가 아래 lot의 익절 LOC 취소 → MOC 매도")
    target = stop.add_mutually_exclusive_group(required=True)
    target.add_argument("--account", help="config/trader/<id>.json 의 id")
    target.add_argument("--all", action="store_true", help="config/trader 의 모든 계좌")
    stop.add_argument("--auto", action="store_true", help="묻지 않음: auto_trade 가 켜진 계좌만 전송 (스케줄 실행용)")
    stop.add_argument("--force", action="store_true", help="손절 확인 시각 전에도 실행 (장중, 주문 마감 전만)")
    report = sub.add_parser("report", help="마감 후: 오늘 낸 주문이 종가대로 체결됐는지, lot별 결과 (읽기 전용)")
    target = report.add_mutually_exclusive_group(required=True)
    target.add_argument("--account", help="config/trader/<id>.json 의 id")
    target.add_argument("--all", action="store_true", help="config/trader 의 모든 계좌")
    report.add_argument("--auto", action="store_true", help="스케줄 실행: 결과를 디스코드로 보냄")
    data = sub.add_parser("data", help="data/ 일봉 갱신: Yahoo + 증권사 전일 종가 (정규장 중)")
    data.add_argument("--account", required=True, help="시세를 물을 계좌 (config/trader/<id>.json 의 id)")
    data.add_argument("--push", action="store_true", help="GitHub의 data/를 받아 갱신하고 커밋 (DONGPA_DATA_REPO, GITHUB_DATA_TOKEN)")
    data.add_argument("--force", action="store_true", help="정규장 밖에서도 실행 (증권사 전일 종가가 하루 전일 수 있음)")
    vr = sub.add_parser("vr", help="VR 사이클 매수·매도 사다리를 증권사 기간잔량 예약으로 (y로 승인)")
    vr.add_argument("--vr", default="main", help="config/vr/<id>.json 의 id")
    vr.add_argument("--credentials", required=True, help="증권사 키 이름 앞부분 (MERITZ_VR → MERITZ_VR_APP_KEY)")
    vr.add_argument("--limit", type=int, help="앞에서 N건만 보냄 (시험용)")
    sub.add_parser("brokers", help="지원하는 증권사")
    args = parser.parse_args(argv)
    load_repo_env()  # broker keys and DONGPA_TRADER_RUNS from .env
    # Only the scheduled runs alert: nobody is watching them. A person at the terminal reads it there.
    scheduled = getattr(args, "auto", False) or (args.command == "data" and args.push)
    records: list[dict] = []
    crash: list[dict] = []
    try:
        return _dispatch(args, records)
    except Exception as exc:
        crash.append(crash_embed(args.command, exc))
        raise
    finally:
        if scheduled:
            send(build_embeds(args.command, records) + crash)


def _dispatch(args: argparse.Namespace, records: list[dict]) -> int:
    if args.command == "brokers":
        print("\n".join(sorted(BROKERS)))
        return 0

    if args.command == "data":
        result = run_data(args.account, force=args.force, push=args.push)
        print(f"[data] {MARKS.get(result['status'], '?')} {result['status']} — {result['message']}")
        for ticker, close in result.get("broker_closes", {}).items():
            print(f"   {ticker} {result['close_of']} 증권사 종가 ${close}")
        if "changed" in result:
            print(f"   바뀐 파일: {', '.join(result['changed']) or '없음'}")
        if result.get("commit"):
            print(f"   커밋: {result['commit']}")
        records.append(result)
        return EXIT.get(result["status"], 3)

    if args.command == "vr":
        record = run_vr_reserve(args.vr, args.credentials, approve=_ask_vr, limit=args.limit)
        print(f"[VR {args.vr}] {MARKS.get(record['status'], '?')} {record['status']} — {record['message']}")
        for item in record.get("sent", []):
            mark = f"예약번호 {item['reservation_id']}" if "reservation_id" in item else f"✗ {item['error']}"
            print(f"   {describe(OrderRequest(**item['request']))}  {mark}")
        print(f"   기록: {record['path']}")
        return EXIT.get(record["status"], 3)

    ids = trader_account_ids() if args.all else [args.account]
    if not ids:
        print("config/trader/*.json 이 없습니다")
        return 1
    worst = 0

    if args.command in ("orders", "stop"):
        for account_id in ids:
            if args.command == "orders":
                record = run_orders(account_id, approve=None if args.auto else _ask)
            else:
                record = run_stop(account_id, approve=None if args.auto else _ask_stop, force=args.force)
            if record["status"] == "halted" and record.get("checks"):
                _print_check(record)
            else:
                _print_orders(record)
            records.append(record)
            worst = max(worst, EXIT.get(record["status"], 3))
        return worst

    if args.command == "report":
        for account_id in ids:
            record = run_report(account_id)
            _print_report(record)
            records.append(record)
            worst = max(worst, EXIT.get(record["status"], 3))
        return worst

    for account_id in ids:
        record = run_check(account_id, force=args.force)
        if args.json:
            print(json.dumps(record, ensure_ascii=False, indent=2, default=str))
        else:
            _print_check(record)
        worst = max(worst, EXIT.get(record["status"], 3))
    return worst
