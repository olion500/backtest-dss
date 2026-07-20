"""Pure business logic for the LOC order book page.

All functions here are free of Streamlit dependencies. They take plain
data (dicts, DataFrames, floats) and return plain data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


import pandas as pd


# --------------- small helpers ---------------

MODE_ALIASES = {
    "안전": "defense",
    "defense": "defense",
    "공세": "offense",
    "offense": "offense",
}


def safe_int(value: object) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, float) and math.isnan(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_mode(value: object) -> str:
    """Map journal mode labels (Korean or English) to canonical keys."""
    if isinstance(value, str):
        trimmed = value.strip()
        lowered = trimmed.lower()
        if lowered in MODE_ALIASES:
            return MODE_ALIASES[lowered]
        if trimmed in MODE_ALIASES:
            return MODE_ALIASES[trimmed]
    return "defense"


# --------------- state extraction ---------------

@dataclass
class BacktestState:
    """Snapshot of the backtest at the last journal row."""
    last_date: object  # date
    current_mode: str  # "defense" | "offense"
    current_cash: float
    current_position_qty: int
    prev_close: float | None
    tranche_budget: float | None
    rsi_value: float | None


def extract_state(
    journal: pd.DataFrame,
    indicators,  # dongpa_engine.Indicators
    ui_values: dict,
    init_cash: float,
) -> BacktestState:
    """Extract current backtest state from the last journal row."""
    from engines.dongpa_engine import _scalar

    journal = journal.copy()
    journal["거래일자"] = pd.to_datetime(journal["거래일자"], errors="coerce")
    last_row = journal.iloc[-1]
    last_date = last_row["거래일자"].date()
    last_timestamp = pd.Timestamp(last_date)

    current_mode = normalize_mode(last_row.get("모드", "안전"))
    current_cash = safe_float(last_row.get("현금")) or init_cash
    current_position_qty = safe_int(last_row.get("보유수량"))
    prev_close = safe_float(last_row.get("종가"))

    # Compute next-day tranche budget from tranche_base_cash (end-of-day value).
    _tranche_base = safe_float(last_row.get("트렌치기반현금"))
    if _tranche_base and _tranche_base > 0:
        _next_slices = (
            int(ui_values["defense_slices"])
            if current_mode == "defense"
            else int(ui_values["offense_slices"])
        )
        tranche_budget = _tranche_base / max(1, _next_slices)
    else:
        tranche_budget = safe_float(last_row.get("일일트렌치예산"))

    # Get RSI value
    rsi_value = None
    if indicators.daily_rsi is not None and last_timestamp in indicators.daily_rsi.index:
        rsi_raw = _scalar(indicators.daily_rsi.loc[last_timestamp])
        if rsi_raw is not None and not pd.isna(rsi_raw):
            rsi_value = float(rsi_raw)

    return BacktestState(
        last_date=last_date,
        current_mode=current_mode,
        current_cash=current_cash,
        current_position_qty=current_position_qty,
        prev_close=prev_close,
        tranche_budget=tranche_budget,
        rsi_value=rsi_value,
    )


# --------------- holdings ---------------

def build_holdings(
    open_trades: pd.DataFrame,
    prev_close: float | None,
) -> list[dict]:
    """Build unrealised P&L rows for currently-open positions."""
    if open_trades.empty or not prev_close:
        return []

    holdings: list[dict] = []
    for _, trade in open_trades.iterrows():
        buy_date = trade.get("매수일자", "")
        buy_price = safe_float(trade.get("매수체결가"))
        buy_qty = safe_int(trade.get("매수수량", 0))
        tp_price = safe_float(trade.get("TP목표가"))
        sl_price = safe_float(trade.get("SL목표가"))
        max_hold = safe_int(trade.get("최대보유일", 0))

        if buy_qty <= 0:
            continue

        current_value = prev_close * buy_qty
        cost_basis = buy_price * buy_qty if buy_price else 0
        unrealized_pnl = current_value - cost_basis
        unrealized_pct = ((prev_close / buy_price) - 1) * 100 if buy_price else None

        hold_period = safe_int(trade.get("보유기간(일)", 0))
        days_left = None
        if max_hold > 0 and hold_period > 0:
            days_left = max_hold - hold_period

        status: list[str] = []
        if tp_price and prev_close >= tp_price:
            status.append("TP도달")
        if sl_price and prev_close <= sl_price:
            status.append("SL도달")
        if days_left is not None and days_left <= 1:
            status.append("만료")

        holdings.append({
            "매수일자": buy_date,
            "매수가": f"${buy_price:.2f}" if buy_price else "-",
            "수량": buy_qty,
            "현재가": f"${prev_close:.2f}",
            "평가손익": f"${unrealized_pnl:.2f}",
            "수익률": f"{unrealized_pct:.1f}%" if unrealized_pct is not None else "-",
            "TP": f"${tp_price:.2f}" if tp_price else "-",
            "SL": f"${sl_price:.2f}" if sl_price else "-",
            "잔여일": days_left if days_left is not None else "-",
            "상태": ", ".join(status) if status else "보유중",
        })

    return holdings


# --------------- order sheet ---------------

def build_order_sheet(
    open_trades: pd.DataFrame,
    prev_close: float | None,
    current_cash: float,
    tranche_budget: float | None,
    current_mode: str,
    ui_values: dict,
    allow_fractional: bool,
) -> tuple[list[dict], list[dict], dict | None]:
    """Build sell (TP/SL/expiry) + buy orders.

    Returns (order_sheet, sl_order_sheet, spread_ctx).
    ``spread_ctx`` is passed to :func:`build_spread_orders` later
    (after netting decides the reference price).
    """
    order_sheet: list[dict] = []
    sl_order_sheet: list[dict] = []

    # --- sell orders ---
    if not open_trades.empty and prev_close:
        for _, trade in open_trades.iterrows():
            buy_date = trade.get("매수일자", "")
            buy_price = safe_float(trade.get("매수체결가"))
            buy_qty = safe_int(trade.get("매수수량", 0))
            tp_price = safe_float(trade.get("TP목표가"))
            sl_price = safe_float(trade.get("SL목표가"))
            max_hold = safe_int(trade.get("최대보유일", 0))

            if buy_qty <= 0:
                continue

            hold_period = safe_int(trade.get("보유기간(일)", 0))
            days_left = None
            is_expiring = False
            if max_hold > 0 and hold_period > 0:
                days_left = max_hold - hold_period
                # The order sheet is for the NEXT trading day, so the lot
                # will have been held for hold_period+1 days.  The engine
                # sells when days+1 >= max_hold, i.e. hold_period+1 >= max_hold,
                # i.e. days_left <= 1.
                is_expiring = days_left <= 1

            # An expiring lot is force-closed at the close regardless of price,
            # so a separate TP/SL order for the same shares would double-count
            # them (both LOC orders could fill on the same close).
            if tp_price and tp_price > 0 and not is_expiring:
                tp_change = ((tp_price / buy_price) - 1) * 100 if buy_price else None
                order_sheet.append({
                    "구분": "매도 (TP)",
                    "주문가": tp_price,
                    "수량": buy_qty,
                    "변화율": f"{tp_change:+.1f}%" if tp_change is not None else "-",
                    "비고": f"매수일: {buy_date}, 매수가: ${buy_price:.2f}" if buy_price else "",
                })

            if sl_price and sl_price > 0 and not is_expiring:
                sl_change = ((sl_price / buy_price) - 1) * 100 if buy_price else None
                sl_order_sheet.append({
                    "구분": "매도 (SL)",
                    "주문가": sl_price,
                    "수량": buy_qty,
                    "변화율": f"{sl_change:+.1f}%" if sl_change is not None else "-",
                    "비고": f"매수일: {buy_date}, 매수가: ${buy_price:.2f}" if buy_price else "",
                })

            if is_expiring:
                order_sheet.append({
                    "구분": "매도 (만료)",
                    "주문가": prev_close,
                    "수량": buy_qty,
                    "변화율": f"{((prev_close / buy_price) - 1) * 100:+.1f}%" if buy_price else "-",
                    "비고": f"잔여일: {days_left}일",
                })

    # --- buy order ---
    spread_ctx: dict | None = None
    if current_cash > 0 and tranche_budget and tranche_budget > 0:
        mode_params_pct = (
            ui_values["defense_buy"] if current_mode == "defense" else ui_values["offense_buy"]
        )
        buy_limit_price = prev_close * (1 + mode_params_pct / 100) if prev_close else None

        if buy_limit_price and buy_limit_price > 0:
            effective_budget = min(tranche_budget, current_cash)
            tp_pct = ui_values["defense_tp"] if current_mode == "defense" else ui_values["offense_tp"]
            sl_pct = ui_values["defense_sl"] if current_mode == "defense" else ui_values["offense_sl"]

            if allow_fractional:
                base_qty = effective_budget / buy_limit_price
            else:
                base_qty = int(effective_budget // buy_limit_price)

            if base_qty > 0:
                new_tp = buy_limit_price * (1 + tp_pct / 100)
                new_sl = buy_limit_price * (1 - sl_pct / 100) if sl_pct > 0 else None

                order_sheet.append({
                    "구분": "매수",
                    "주문가": buy_limit_price,
                    "수량": base_qty,
                    "변화율": f"{mode_params_pct:+.1f}%",
                    "비고": (
                        f"→ TP: ${new_tp:.2f}, SL: ${new_sl:.2f}"
                        if new_sl
                        else f"→ TP: ${new_tp:.2f}"
                    ),
                })

                spread_ctx = {
                    "buy_limit_price": buy_limit_price,
                    "effective_budget": effective_budget,
                    "base_qty": base_qty,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                }

    return order_sheet, sl_order_sheet, spread_ctx


# --------------- netting ---------------

@dataclass
class NettingResult:
    order_sheet: list[dict]
    netting_msg: str
    netting_details: list[dict]
    netting_floor_price: float | None
    scenario_rows: list[dict] = field(default_factory=list)


def apply_netting(
    order_sheet: list[dict],
    prev_close: float | None,
    allow_fractional: bool,
) -> NettingResult:
    """Offset matching sell/buy quantities in-place.

    Netting consumes the lowest-priced sells first.  Each consumed sell
    at price ``p`` comes back as a mirror buy at ``p - 0.01`` (if the
    close lands below ``p`` that sell would not have filled, so the
    netted-away buy must happen), and the consumed buy quantity comes
    back as a mirror sell at ``buy_price + 0.01`` (if the close lands
    above the buy limit the buy would not have filled, so the netted-away
    sells must happen).  The resulting sheet reproduces the exact net
    outcome of the original orders for every closing price, without ever
    double-counting shares.

    Per-closing-range outcomes are returned in ``scenario_rows`` — they
    are informational only, NOT placeable orders.

    Returns a new ``NettingResult`` with the cleaned order sheet.
    """
    netting_msg = ""
    netting_details: list[dict] = []
    scenario_rows: list[dict] = []
    netting_floor_price: float | None = None

    sell_indices = [i for i, r in enumerate(order_sheet) if r["구분"].startswith("매도")]
    buy_index = next((i for i, r in enumerate(order_sheet) if r["구분"] == "매수"), None)

    if buy_index is None or not sell_indices:
        return NettingResult(order_sheet, netting_msg, netting_details, netting_floor_price)

    buy_price = float(order_sheet[buy_index]["주문가"])
    total_buy_qty = float(order_sheet[buy_index]["수량"])
    fmt_qty = (lambda q: f"{q:,.4f}") if allow_fractional else (lambda q: f"{int(q):,}")

    # Only net sell orders where sell_price <= buy_price
    nettable_sell_indices: list[int] = []
    non_nettable_sell_indices: list[int] = []
    for i in sell_indices:
        sell_price = float(order_sheet[i]["주문가"])
        if sell_price <= buy_price:
            nettable_sell_indices.append(i)
        else:
            non_nettable_sell_indices.append(i)

    nettable_sell_qty = sum(float(order_sheet[i]["수량"]) for i in nettable_sell_indices)

    if nettable_sell_qty <= 0 or total_buy_qty <= 0:
        return NettingResult(order_sheet, netting_msg, netting_details, netting_floor_price)

    offset = min(nettable_sell_qty, total_buy_qty)

    # Pre-netting sell groups (price → qty) for the scenario table.
    sell_groups: dict[float, float] = defaultdict(float)
    for i in nettable_sell_indices:
        sell_groups[float(order_sheet[i]["주문가"])] += float(order_sheet[i]["수량"])
    sorted_sells = sorted(sell_groups.items())

    # --- consume the cheapest sells first ---
    # Bottom-up consumption is required for the mirror rows below to
    # reproduce the correct net outcome in every closing-price range.
    consumed_by_price: dict[float, float] = defaultdict(float)
    consumed_amt = 0.0
    remaining = offset
    for i in sorted(nettable_sell_indices, key=lambda idx: float(order_sheet[idx]["주문가"])):
        if remaining <= 0:
            break
        row = order_sheet[i]
        row_qty = float(row["수량"])
        row_price = float(row["주문가"])
        reduction = min(row_qty, remaining)
        remaining -= reduction
        consumed_by_price[row_price] += reduction
        consumed_amt += reduction * row_price
        netting_details.append({
            "매도": row["구분"],
            "매도가": row_price,
            "매수가": buy_price,
            "상쇄 수량": reduction,
            "사유": f"매도가 ${row_price:.2f} ≤ 매수가 ${buy_price:.2f}",
        })
        new_qty = row_qty - reduction
        if not allow_fractional:
            new_qty = int(new_qty)
        if new_qty > 0:
            row["수량"] = new_qty
            prior_note = row.get("비고", "")
            note = f"퉁치기 {fmt_qty(reduction)}주 상쇄"
            row["비고"] = f"{prior_note} | {note}" if prior_note else note
        else:
            order_sheet[i] = None  # type: ignore[assignment]

    # --- reduce the buy row ---
    net_buy = total_buy_qty - offset
    if not allow_fractional:
        net_buy = int(net_buy)
    if net_buy > 0:
        order_sheet[buy_index]["수량"] = net_buy
        order_sheet[buy_index]["비고"] = f"퉁치기 후 순매수 ({fmt_qty(offset)}주 상쇄)"
    else:
        order_sheet[buy_index] = None  # type: ignore[assignment]

    cash_impact = consumed_amt - buy_price * offset
    cash_str = (
        f"순 유입 ${cash_impact:,.2f}" if cash_impact >= 0 else f"순 유출 ${-cash_impact:,.2f}"
    )

    net_sell = nettable_sell_qty - offset
    if net_buy > 0:
        netting_msg = (
            f"퉁치기 적용: 매도 {fmt_qty(offset)}주 상쇄 → "
            f"순매수 {fmt_qty(net_buy)}주 ({cash_str})"
        )
    elif net_sell > 0:
        netting_msg = (
            f"퉁치기 적용: 매수 {fmt_qty(total_buy_qty)}주 상쇄 → "
            f"순매도 {fmt_qty(net_sell)}주 ({cash_str})"
        )
    else:
        netting_msg = (
            f"퉁치기 적용: 매수·매도 {fmt_qty(total_buy_qty)}주 완전상쇄 ({cash_str})"
        )

    if non_nettable_sell_indices:
        non_nettable_qty = sum(
            float(order_sheet[i]["수량"])
            for i in non_nettable_sell_indices
            if order_sheet[i] is not None
        )
        if non_nettable_qty > 0:
            netting_msg += f" | 퉁치기 불가 매도 {fmt_qty(non_nettable_qty)}주 (매도가 > 매수가)"

    order_sheet = [r for r in order_sheet if r is not None]

    # --- mirror rows ---
    # Consumed sell at price p → buy back just below p (sell would not fill).
    for price in sorted(consumed_by_price):
        qty = consumed_by_price[price]
        if not allow_fractional:
            qty = int(qty)
        if qty <= 0:
            continue
        mirror_price = price - 0.01
        pct = ((mirror_price / prev_close) - 1) * 100 if prev_close else 0
        order_sheet.append({
            "구분": "매수",
            "주문가": mirror_price,
            "수량": qty,
            "변화율": f"{pct:+.1f}%",
            "비고": f"종가 < ${price:.2f} 시 매도미체결 → 매수",
        })
    if consumed_by_price:
        netting_floor_price = min(consumed_by_price) - 0.01

    # Consumed buy quantity → sell back just above the buy limit (buy would not fill).
    ceiling_qty = offset if allow_fractional else int(offset)
    ceiling_price = buy_price + 0.01
    pct = ((ceiling_price / prev_close) - 1) * 100 if prev_close else 0
    order_sheet.append({
        "구분": "매도",
        "주문가": ceiling_price,
        "수량": ceiling_qty,
        "변화율": f"{pct:+.1f}%",
        "비고": f"종가 > ${buy_price:.2f} 시 매수미체결 → 매도",
    })

    # --- per-closing-range outcome table (informational only) ---
    def _outcome(net: float) -> str:
        qty = abs(net)
        if net > 0:
            return f"순매수 {fmt_qty(qty)}주"
        if net < 0:
            return f"순매도 {fmt_qty(qty)}주"
        return "±0주"

    first_price = sorted_sells[0][0]
    scenario_rows.append({
        "종가 구간": f"< ${first_price:.2f}",
        "순결과": _outcome(total_buy_qty),
        "설명": "매도 전량 미체결, 매수만 체결",
    })
    cum_sell = 0.0
    for idx, (sp, sq) in enumerate(sorted_sells):
        cum_sell += sq
        upper = sorted_sells[idx + 1][0] if idx + 1 < len(sorted_sells) else buy_price
        scenario_rows.append({
            "종가 구간": f"${sp:.2f} ~ ${upper:.2f}",
            "순결과": _outcome(total_buy_qty - cum_sell),
            "설명": f"매도 {fmt_qty(cum_sell)}주 + 매수 {fmt_qty(total_buy_qty)}주 체결",
        })
    scenario_rows.append({
        "종가 구간": f"> ${buy_price:.2f}",
        "순결과": _outcome(-nettable_sell_qty),
        "설명": "매수 미체결, 매도 전량 체결",
    })

    return NettingResult(
        order_sheet, netting_msg, netting_details, netting_floor_price, scenario_rows,
    )


# --------------- spread buy orders ---------------

def build_spread_orders(
    spread_ctx: dict,
    netting_floor_price: float | None,
    prev_close: float | None,
    ui_values: dict,
    allow_fractional: bool,
) -> list[dict]:
    """Generate lower-price spread buy orders below the base buy."""
    ref_price = netting_floor_price if netting_floor_price else spread_ctx["buy_limit_price"]
    eff_budget = spread_ctx["effective_budget"]
    s_tp_pct = spread_ctx["tp_pct"]
    s_sl_pct = spread_ctx["sl_pct"]

    if allow_fractional:
        ref_qty = eff_budget / ref_price
    else:
        ref_qty = int(eff_budget // ref_price)

    max_spread = ui_values.get("spread_buy_levels", 5)
    s_step = ui_values.get("spread_buy_step", 1)
    min_drop_pct = -50.0

    rows: list[dict] = []
    if ref_qty <= 0:
        return rows

    for n in range(1, max_spread + 1):
        incr = n * s_step
        sp_price = eff_budget / (ref_qty + incr)

        drop = ((sp_price / ref_price) - 1) * 100
        if drop < min_drop_pct:
            break

        sp_tp = sp_price * (1 + s_tp_pct / 100)
        sp_sl = sp_price * (1 - s_sl_pct / 100) if s_sl_pct > 0 else None
        pct = ((sp_price / prev_close) - 1) * 100 if prev_close else 0

        note = f"TP: ${sp_tp:.2f}"
        if sp_sl:
            note += f", SL: ${sp_sl:.2f}"

        rows.append({
            "구분": f"매수 (+{incr}주)",
            "주문가": sp_price,
            "수량": s_step,
            "변화율": f"{pct:+.1f}%",
            "비고": note,
        })

    return rows
