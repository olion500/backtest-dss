"""Tests for order_book_engine pure functions."""

from datetime import date

import pandas as pd
import pytest

from engines.order_book_engine import (
    MOC_SELL,
    BacktestState,
    NettingResult,
    apply_netting,
    build_holdings,
    build_order_sheet,
    build_stop_loss_orders,
    build_spread_orders,
    normalize_mode,
    safe_float,
    safe_int,
)


# --------------- helpers ---------------

class TestSafeInt:
    def test_normal(self):
        assert safe_int(5) == 5

    def test_float(self):
        assert safe_int(5.9) == 5

    def test_none(self):
        assert safe_int(None) == 0

    def test_nan(self):
        assert safe_int(float("nan")) == 0

    def test_string(self):
        assert safe_int("bad") == 0


class TestSafeFloat:
    def test_normal(self):
        assert safe_float(3.14) == 3.14

    def test_none(self):
        assert safe_float(None) is None

    def test_nan(self):
        assert safe_float(float("nan")) is None


class TestNormalizeMode:
    def test_korean_defense(self):
        assert normalize_mode("안전") == "defense"

    def test_korean_offense(self):
        assert normalize_mode("공세") == "offense"

    def test_english(self):
        assert normalize_mode("offense") == "offense"

    def test_unknown(self):
        assert normalize_mode("unknown") == "defense"

    def test_non_string(self):
        assert normalize_mode(42) == "defense"


# --------------- build_holdings ---------------

def _make_open_trades(entries: list[dict]) -> pd.DataFrame:
    """Create a simple open_trades DataFrame."""
    return pd.DataFrame(entries)


class TestBuildHoldings:
    def test_empty_trades(self):
        assert build_holdings(pd.DataFrame(), 100.0) == []

    def test_no_prev_close(self):
        trades = _make_open_trades([{"매수체결가": 50, "매수수량": 10, "TP목표가": 55, "SL목표가": 45, "최대보유일": 7, "보유기간(일)": 3}])
        assert build_holdings(trades, None) == []

    def test_single_holding(self):
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": 90.0,
            "최대보유일": 7,
            "보유기간(일)": 3,
        }])
        holdings = build_holdings(trades, 105.0)
        assert len(holdings) == 1
        h = holdings[0]
        assert h["수량"] == 10
        assert h["상태"] == "보유중"

    def test_tp_reached(self):
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 5,
            "TP목표가": 110.0,
            "SL목표가": None,
            "최대보유일": 30,
            "보유기간(일)": 2,
        }])
        holdings = build_holdings(trades, 112.0)
        assert "TP도달" in holdings[0]["상태"]

    def test_expired(self):
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 5,
            "TP목표가": 110.0,
            "SL목표가": None,
            "최대보유일": 5,
            "보유기간(일)": 5,
        }])
        holdings = build_holdings(trades, 102.0)
        assert "만료" in holdings[0]["상태"]

    def test_expiring_next_day(self):
        """Position expiring on the next trading day (days_left=1) should show 만료."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 5,
            "TP목표가": 110.0,
            "SL목표가": None,
            "최대보유일": 7,
            "보유기간(일)": 6,
        }])
        holdings = build_holdings(trades, 102.0)
        assert "만료" in holdings[0]["상태"]


# --------------- build_stop_loss_orders ---------------

class TestBuildStopLossOrders:
    _LOG = pd.DataFrame([
        {"매수일자": "2024-01-02", "매도일자": "2024-01-10", "매도평균": 88.5, "매도수량": 10.0, "SL목표가": 90.0, "청산사유": "SL", "상태": "완료"},
        {"매수일자": "2024-01-03", "매도일자": "2024-01-10", "매도평균": 88.5, "매도수량": 7.0, "SL목표가": None, "청산사유": "TP", "상태": "완료"},
        {"매수일자": "2024-01-04", "매도일자": "2024-01-09", "매도평균": 89.0, "매도수량": 5.0, "SL목표가": 92.0, "청산사유": "SL", "상태": "완료"},
    ])

    def test_lists_only_last_day_stop_losses_as_moc(self):
        rows = build_stop_loss_orders(self._LOG, date(2024, 1, 10))
        assert [(r["구분"], r["수량"], r["주문가"]) for r in rows] == [("매도 (손절)", 10, 88.5)]
        assert "MOC" in rows[0]["비고"]

    def test_empty_log(self):
        assert build_stop_loss_orders(pd.DataFrame(), date(2024, 1, 10)) == []


# --------------- build_order_sheet ---------------

_DEFAULT_UI = {
    "defense_buy": 3.0,
    "defense_tp": 0.2,
    "defense_sl": 0.0,
    "offense_buy": 5.0,
    "offense_tp": 2.5,
    "offense_sl": 0.0,
    "spread_buy_levels": 3,
    "spread_buy_step": 1,
}


class TestBuildOrderSheet:
    def test_cash_limited_caps_budget_at_cash(self):
        ui = {**_DEFAULT_UI, "cash_limited_buy": True}
        sheet, ctx = build_order_sheet(pd.DataFrame(), 100.0, 500.0, 1000.0, "defense", ui, False)
        buy = next(r for r in sheet if r["구분"] == "매수")
        assert ctx["effective_budget"] == 500.0
        assert buy["수량"] == 4  # 500 // 103
        assert "현금 부족" not in buy["비고"]

    def test_not_cash_limited_uses_full_tranche_and_warns(self):
        ui = {**_DEFAULT_UI, "cash_limited_buy": False}
        sheet, ctx = build_order_sheet(pd.DataFrame(), 100.0, 950.0, 1000.0, "defense", ui, False)
        buy = next(r for r in sheet if r["구분"] == "매수")
        assert ctx["effective_budget"] == 1000.0
        assert buy["수량"] == 9  # 1000 // 103 → 927 <= 950 cash
        assert "현금 부족" in buy["비고"]

    def test_not_cash_limited_skips_buy_cash_cannot_cover(self):
        """9 × 103 = 927 > 500 cash: the backtest would skip the buy, so no order."""
        ui = {**_DEFAULT_UI, "cash_limited_buy": False}
        sheet, ctx = build_order_sheet(pd.DataFrame(), 100.0, 500.0, 1000.0, "defense", ui, False)
        assert not any(r["구분"] == "매수" for r in sheet)
        assert ctx is None

    def test_cash_exactly_covering_the_order_keeps_it(self):
        """9 × 103 = 927 = cash: the backtest's ``trade_value <= cash`` still buys."""
        sheet, _ = build_order_sheet(pd.DataFrame(), 100.0, 927.0, 1000.0, "defense", _DEFAULT_UI, False)
        assert [r["수량"] for r in sheet if r["구분"] == "매수"] == [9]

    def test_skipped_buy_leaves_sells_unnetted(self):
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 98.0,
            "매수수량": 10,
            "TP목표가": 101.0,
            "SL목표가": None,
            "최대보유일": 30,
            "보유기간(일)": 3,
        }])
        sheet, ctx = build_order_sheet(trades, 100.0, 50.0, 1000.0, "defense", _DEFAULT_UI, False)
        result = apply_netting(sheet, 100.0, False)
        assert ctx is None
        assert [(r["구분"], r["수량"]) for r in result.order_sheet] == [("매도 (익절)", 10)]

    def test_buy_only_no_positions(self):
        """No open positions, only a buy order is generated."""
        sheet, ctx = build_order_sheet(
            pd.DataFrame(), 100.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        buy_rows = [r for r in sheet if r["구분"] == "매수"]
        assert len(buy_rows) == 1
        # defense: limit = 100 × 1.03, qty = 1000 // 103
        assert buy_rows[0]["주문가"] == pytest.approx(103.0)
        assert buy_rows[0]["수량"] == 9
        assert ctx["base_qty"] == 9 and ctx["tp_pct"] == 0.2

    def test_offense_buy_uses_offense_params(self):
        sheet, ctx = build_order_sheet(
            pd.DataFrame(), 100.0, 5000.0, 1000.0, "offense", {**_DEFAULT_UI, "offense_sl": 4.0}, False,
        )
        buy = next(r for r in sheet if r["구분"] == "매수")
        assert buy["주문가"] == pytest.approx(105.0)
        assert buy["수량"] == 9  # 1000 // 105
        assert ctx["tp_pct"] == 2.5 and ctx["sl_pct"] == 4.0
        # TP / SL are quoted off the limit: 105 × 1.025, 105 × 0.96
        assert "익절: $107.62" in buy["비고"] and "손절: $100.80" in buy["비고"]

    def test_sell_tp_generated(self):
        """Open position generates a TP sell order."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 95.0,
            "매수수량": 10,
            "TP목표가": 100.0,
            "SL목표가": None,
            "최대보유일": 30,
            "보유기간(일)": 3,
        }])
        sheet, ctx = build_order_sheet(
            trades, 98.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        tp_rows = [r for r in sheet if "익절" in r["구분"]]
        assert [(r["주문가"], r["수량"]) for r in tp_rows] == [(100.0, 10)]
        assert not any(r["구분"] == MOC_SELL for r in sheet)

    def test_sell_rows_carry_the_lots_levels(self):
        """TP and expiry sells carry the lot's buy/TP/SL prices for the sheet's progress gauge."""
        trades = _make_open_trades([
            {"매수일자": "2024-01-02", "매수체결가": 100.0, "매수수량": 10, "TP목표가": 110.0,
             "SL목표가": 90.0, "최대보유일": 30, "보유기간(일)": 3},
            {"매수일자": "2024-01-03", "매수체결가": 96.0, "매수수량": 5, "TP목표가": 99.0,
             "SL목표가": None, "최대보유일": 4, "보유기간(일)": 3},
        ])
        sheet, _ = build_order_sheet(trades, 98.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False)
        tp = next(r for r in sheet if r["구분"] == "매도 (익절)")
        moc = next(r for r in sheet if r["구분"] == MOC_SELL)
        assert (tp["매수가"], tp["익절가"], tp["손절가"]) == (100.0, 110.0, 90.0)
        assert (moc["매수가"], moc["익절가"], moc["손절가"]) == (96.0, 99.0, None)
        # display-only keys survive netting untouched
        result = apply_netting(sheet, 98.0, False)
        assert all(r.get("익절가") == 110.0 for r in result.order_sheet if r["구분"] == "매도 (익절)")

    def test_open_lot_gets_no_stop_loss_row(self):
        """An open lot's SL is unknown until the close, so it is not on the sheet."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": 90.0,
            "최대보유일": 30,
            "보유기간(일)": 3,
        }])
        sheet, _ = build_order_sheet(
            trades, 98.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        assert not any("손절" in r["구분"] for r in sheet)

    def test_expiring_sell(self):
        """Position at max hold generates a 만료 sell."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": None,
            "최대보유일": 5,
            "보유기간(일)": 5,
        }])
        sheet, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        expiry_rows = [r for r in sheet if r["구분"] == "매도 (MOC)"]
        assert len(expiry_rows) == 1
        assert expiry_rows[0]["수량"] == 10
        assert expiry_rows[0]["주문가"] == 102.0

    def test_two_days_left_is_not_expiring(self):
        """Held 5 of 7 days: tomorrow is day 6 < 7, so the TP sell stays and no MOC."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02", "매수체결가": 100.0, "매수수량": 10, "TP목표가": 110.0,
            "SL목표가": None, "최대보유일": 7, "보유기간(일)": 5,
        }])
        sheet, _ = build_order_sheet(trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False)
        assert [r["구분"] for r in sheet if r["구분"].startswith("매도")] == ["매도 (익절)"]

    def test_expiring_next_day_sell(self):
        """Position expiring on next trading day (days_left=1) generates a 만료 sell."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": None,
            "최대보유일": 7,
            "보유기간(일)": 6,
        }])
        sheet, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        expiry_rows = [r for r in sheet if r["구분"] == "매도 (MOC)"]
        assert len(expiry_rows) == 1

    def test_expiring_lot_has_no_tp_or_sl(self):
        """Expiring lot must not emit TP/SL rows — the 만료 sell already
        covers those shares (both would double-count on the same close)."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": 90.0,
            "최대보유일": 7,
            "보유기간(일)": 6,
        }])
        sheet, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        assert [r for r in sheet if "익절" in r["구분"]] == []
        expiry_rows = [r for r in sheet if r["구분"] == "매도 (MOC)"]
        assert len(expiry_rows) == 1
        assert expiry_rows[0]["수량"] == 10

    def test_no_cash_no_buy(self):
        """No cash → no buy order."""
        sheet, ctx = build_order_sheet(
            pd.DataFrame(), 100.0, 0.0, 0.0, "defense", _DEFAULT_UI, False,
        )
        buy_rows = [r for r in sheet if r["구분"] == "매수"]
        assert len(buy_rows) == 0
        assert ctx is None


# --------------- apply_netting ---------------

class TestApplyNetting:
    def test_no_netting_without_overlap(self):
        """Sell price > buy price → no netting."""
        sheet = [
            {"구분": "매도 (익절)", "주문가": 110.0, "수량": 10, "변화율": "+10%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 105.0, False)
        assert result.netting_msg == ""
        assert result.netting_details == []
        # Original rows unchanged
        buy_rows = [r for r in result.order_sheet if r["구분"] == "매수"]
        assert len(buy_rows) == 1

    def test_full_netting(self):
        """Equal sell and buy qty → complete offset."""
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 10, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert "완전상쇄" in result.netting_msg
        # The TP sell and the buy cancel out; only the two mirror orders are placed:
        # buy back below the sell price, sell above the buy limit.
        assert [(r["구분"], r["주문가"], r["수량"]) for r in result.order_sheet] == [
            ("매수 (퉁치기)", pytest.approx(97.99), 10),
            ("매도 (퉁치기)", pytest.approx(100.01), 10),
        ]
        assert result.netting_floor_price == pytest.approx(97.99)
        assert [r["순결과"] for r in result.scenario_rows] == ["순매수 10주", "±0주", "순매도 10주"]

    def test_partial_netting_buy_larger(self):
        """Buy qty > sell qty → net buy remains."""
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 5, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert "순매수" in result.netting_msg
        net_buy = [r for r in result.order_sheet if r["구분"] == "매수" and "퉁치기 후" in r.get("비고", "")]
        assert len(net_buy) == 1
        assert net_buy[0]["수량"] == 5

    def test_partial_netting_sell_larger(self):
        """Sell qty > buy qty → net sell remains."""
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 15, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert "순매도 5주" in result.netting_msg
        tp = [r for r in result.order_sheet if r["구분"] == "매도 (익절)"]
        assert [r["수량"] for r in tp] == [5]
        assert not any(r["구분"] == "매수" for r in result.order_sheet)

    def test_netting_produces_scenario_rows(self):
        """Netting should add floor (buy-only) and ceiling (sell-only) scenario rows."""
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 10, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        # Should have floor price row (below sell price) and ceiling row (above buy price)
        assert result.netting_floor_price is not None
        assert result.netting_floor_price < 98.0


def _net_fill(rows: list[dict], close: float) -> float:
    """Simulate LOC fills at a given close: buys fill when close <= limit,
    sells fill when close >= limit. Returns net share change."""
    net = 0.0
    for r in rows:
        price = float(r["주문가"])
        qty = float(r["수량"])
        if r["구분"].startswith("매수"):
            if close <= price + 1e-9:
                net += qty
        elif r["구분"] == "매도 (MOC)":
            net -= qty  # market on close: fills at any close
        elif r["구분"].startswith("매도"):
            if close >= price - 1e-9:
                net -= qty
    return net


class TestNettingConsistency:
    """The netted sheet must reproduce the original net outcome at every
    close, and never list more sell shares than are actually held."""

    # Real-world case: expiring (MOC) lot + two TP lots + buy 20 @ 145.97
    _SHEET = [
        {"구분": "매도 (MOC)", "주문가": 135.47, "수량": 23, "변화율": "-", "비고": "잔여일: 1일"},
        {"구분": "매도 (익절)", "주문가": 144.28, "수량": 23, "변화율": "-", "비고": ""},
        {"구분": "매도 (익절)", "주문가": 137.18, "수량": 22, "변화율": "-", "비고": ""},
        {"구분": "매수", "주문가": 145.97, "수량": 20, "변화율": "-", "비고": ""},
    ]

    def _copy(self):
        import copy
        return copy.deepcopy(self._SHEET)

    def test_net_outcome_preserved_at_every_close(self):
        original = self._copy()
        result = apply_netting(self._copy(), 135.47, False)
        closes = [130.0, 135.46, 135.47, 136.0, 137.17, 137.18, 140.0,
                  144.27, 144.28, 145.0, 145.97, 145.98, 150.0]
        for close in closes:
            assert _net_fill(result.order_sheet, close) == pytest.approx(
                _net_fill(original, close)
            ), f"net outcome mismatch at close={close}"

    def test_total_sell_qty_not_exceed_holdings(self):
        held = 23 + 23 + 22  # shares behind the sell rows
        result = apply_netting(self._copy(), 135.47, False)
        total_sell = sum(
            r["수량"] for r in result.order_sheet if r["구분"].startswith("매도")
        )
        assert total_sell == held

    def test_lowest_priced_sell_consumed_first(self):
        result = apply_netting(self._copy(), 135.47, False)
        assert len(result.netting_details) == 1
        assert result.netting_details[0]["매도가"] == 135.47
        assert result.netting_details[0]["상쇄 수량"] == 20

    def test_multi_group_consumption_buy_larger(self):
        """Buy larger than sells across two price groups → each consumed
        group mirrors back as a buy at its own price - 0.01."""
        import copy
        sheet = [
            {"구분": "매도 (익절)", "주문가": 95.0, "수량": 5, "변화율": "-", "비고": ""},
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 10, "변화율": "-", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 20, "변화율": "-", "비고": ""},
        ]
        original = copy.deepcopy(sheet)
        result = apply_netting(sheet, 99.0, False)
        for close in [90.0, 94.99, 95.0, 96.0, 97.99, 98.0, 99.0, 100.0, 100.01, 105.0]:
            assert _net_fill(result.order_sheet, close) == pytest.approx(
                _net_fill(original, close)
            ), f"net outcome mismatch at close={close}"
        mirror_prices = sorted(
            r["주문가"] for r in result.order_sheet
            if r["구분"] == "매수 (퉁치기)"
        )
        assert mirror_prices == [94.99, 97.99]

    def test_moc_sell_gets_no_mirror_buy(self):
        result = apply_netting(self._copy(), 135.47, False)
        assert not any(r["구분"] == "매수 (퉁치기)" for r in result.order_sheet)
        assert result.netting_floor_price is None

    def test_scenario_rows_are_informational(self):
        result = apply_netting(self._copy(), 135.47, False)
        assert result.scenario_rows
        # ranges: <137.18 (MOC always fills), 2 TP groups, >145.97 → 4 rows
        assert len(result.scenario_rows) == 4
        assert result.scenario_rows[0]["순결과"] == "순매도 3주"  # MOC 23 − buy 20
        assert result.scenario_rows[-1]["순결과"] == "순매도 68주"
        # scenario rows must not leak into the order sheet
        for r in result.order_sheet:
            assert "구간" not in r.get("비고", "")


# --------------- build_spread_orders ---------------

class TestBuildSpreadOrders:
    def test_basic_spread(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "tp_pct": 2.0,
            "sl_pct": 0.0,
        }
        rows = build_spread_orders(ctx, None, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 1}, False)
        # budget 1000 @ 100 → 10 shares; step n is where 1000 buys 10 + n shares
        assert [r["주문가"] for r in rows] == [90.91, 83.33, 76.92]
        assert [r["구분"] for r in rows] == ["매수 (+1주)", "매수 (+2주)", "매수 (+3주)"]
        assert [r["수량"] for r in rows] == [1, 1, 1]
        assert rows[0]["변화율"] == "-9.1%"
        assert rows[0]["비고"] == "익절: $92.73"  # 90.91 × 1.02

    def test_with_netting_floor(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "tp_pct": 2.0,
            "sl_pct": 0.0,
        }
        rows = build_spread_orders(ctx, 89.99, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 1}, False)
        # The ladder hangs from the netting floor: 1000 @ 89.99 → 11 shares, steps at 12, 13, 14
        assert [r["주문가"] for r in rows] == [83.33, 76.92, 71.43]

    def test_ladder_drops_steps_cash_cannot_cover(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "cash": 1000.0,
            "tp_pct": 2.0,
            "sl_pct": 0.0,
        }
        rows = build_spread_orders(ctx, None, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 1}, False)
        full = build_spread_orders({**ctx, "cash": 10_000.0}, None, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 1}, False)
        assert len(full) == 3
        kept = [r for n, r in enumerate(full, start=1) if (10 + n) * r["주문가"] <= 1000.0]
        assert rows == kept
        assert [r["주문가"] for r in rows] == [83.33, 76.92]  # 11 × 90.91 = 1000.01 > cash

    def test_zero_levels(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "tp_pct": 2.0,
            "sl_pct": 0.0,
        }
        rows = build_spread_orders(ctx, None, 100.0, {"spread_buy_levels": 0, "spread_buy_step": 1}, False)
        assert len(rows) == 0

    def test_sl_included_in_note(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "tp_pct": 2.0,
            "sl_pct": 5.0,
        }
        rows = build_spread_orders(ctx, None, 100.0, {"spread_buy_levels": 1, "spread_buy_step": 1}, False)
        assert len(rows) == 1
        assert "손절" in rows[0]["비고"]


# --------------- extract_state ---------------

class TestExtractState:
    @staticmethod
    def _journal(**last):
        row = {"거래일자": "2024-01-05", "모드": "안전", "다음모드": "공세", "현금": 7000.0,
               "보유수량": 30, "종가": 101.5, "트렌치기반현금": 7000.0, "일일트렌치예산": 1000.0}
        row.update(last)
        return pd.DataFrame([{"거래일자": "2024-01-04", "모드": "안전", "다음모드": "안전"}, row])

    @staticmethod
    def _indicators(rsi=None):
        from engines.dongpa_engine import Indicators
        return Indicators(strategy="rsi", daily_rsi=rsi)

    def test_next_session_state_from_last_row(self):
        from engines.order_book_engine import extract_state
        rsi = pd.Series([48.0, 52.5], index=pd.DatetimeIndex(["2024-01-04", "2024-01-05"]))
        state = extract_state(self._journal(), self._indicators(rsi), {"defense_slices": 7, "offense_slices": 4}, 10_000)
        assert state.last_date == date(2024, 1, 5)
        assert state.current_mode == "offense"  # 다음모드, not 모드
        assert state.current_cash == 7000.0
        assert state.current_position_qty == 30
        assert state.prev_close == 101.5
        assert state.tranche_budget == 7000.0 / 4  # tranche base ÷ next mode's slices
        assert state.rsi_value == 52.5

    def test_defense_slices_for_defense_mode(self):
        from engines.order_book_engine import extract_state
        state = extract_state(self._journal(다음모드="안전"), self._indicators(), {"defense_slices": 7, "offense_slices": 4}, 10_000)
        assert state.tranche_budget == 1000.0
        assert state.rsi_value is None

    def test_falls_back_to_daily_budget_without_tranche_base(self):
        from engines.order_book_engine import extract_state
        state = extract_state(self._journal(트렌치기반현금=0.0), self._indicators(), {"defense_slices": 7, "offense_slices": 4}, 10_000)
        assert state.tranche_budget == 1000.0


# --------------- holdings values / netting edges ---------------

def _lot(**overrides):
    lot = {"매수일자": "2024-01-02", "매수체결가": 100.0, "매수수량": 10, "TP목표가": 110.0,
           "SL목표가": 90.0, "최대보유일": 7, "보유기간(일)": 3}
    lot.update(overrides)
    return lot


class TestHoldingValues:
    def test_unrealized_pnl_and_days_left(self):
        h = build_holdings(_make_open_trades([_lot()]), 105.0)[0]
        assert (h["매수가"], h["현재가"], h["평가손익"], h["수익률"]) == ("$100.00", "$105.00", "$50.00", "5.0%")
        assert (h["TP"], h["SL"], h["잔여일"]) == ("$110.00", "$90.00", 4)

    def test_status_boundaries(self):
        assert build_holdings(_make_open_trades([_lot()]), 110.0)[0]["상태"] == "TP도달"
        assert build_holdings(_make_open_trades([_lot()]), 90.0)[0]["상태"] == "SL도달"
        assert build_holdings(_make_open_trades([_lot(**{"보유기간(일)": 5})]), 100.0)[0]["상태"] == "보유중"

    def test_skips_empty_lots(self):
        assert build_holdings(_make_open_trades([_lot(매수수량=0)]), 100.0) == []

    def test_order_sheet_skips_empty_lots(self):
        sheet, _ = build_order_sheet(_make_open_trades([_lot(매수수량=0)]), 100.0, 0.0, None, "defense", _DEFAULT_UI, False)
        assert sheet == []


class TestNettingEdges:
    def test_sell_at_the_buy_price_is_nettable(self):
        sheet = [
            {"구분": "매도 (익절)", "주문가": 100.0, "수량": 4, "변화율": "", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert result.netting_details[0]["상쇄 수량"] == 4
        assert result.netting_msg == "퉁치기 적용: 매도 4주 상쇄 → 순매수 6주 (순 유입 $0.00)"
        assert next(r for r in result.order_sheet if r["구분"] == "매수")["수량"] == 6

    def test_cash_impact_and_unnettable_sells_in_message(self):
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 4, "변화율": "", "비고": ""},
            {"구분": "매도 (익절)", "주문가": 120.0, "수량": 3, "변화율": "", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        # 4 shares sold at 98 instead of bought at 100 → 8 less cash out
        assert result.netting_msg == (
            "퉁치기 적용: 매도 4주 상쇄 → 순매수 6주 (순 유출 $8.00)"
            " | 퉁치기 불가 매도 3주 (매도가 > 매수가)"
        )
        assert [r["수량"] for r in result.order_sheet if r["주문가"] == 120.0] == [3]

    def test_partially_consumed_sell_keeps_the_rest(self):
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 15, "변화율": "", "비고": "lot A"},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        tp = next(r for r in result.order_sheet if r["구분"] == "매도 (익절)")
        assert tp["수량"] == 5
        assert tp["비고"] == "lot A | 퉁치기 10주 상쇄"
        assert result.netting_msg.startswith("퉁치기 적용: 매수 10주 상쇄 → 순매도 5주")

    def test_scenario_table_for_two_sell_groups(self):
        sheet = [
            {"구분": "매도 (익절)", "주문가": 95.0, "수량": 5, "변화율": "", "비고": ""},
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 10, "변화율": "", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 20, "변화율": "", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert [(r["종가 구간"], r["순결과"]) for r in result.scenario_rows] == [
            ("< $95.00", "순매수 20주"),
            ("$95.00 ~ $98.00", "순매수 15주"),
            ("$98.00 ~ $100.00", "순매수 5주"),
            ("> $100.00", "순매도 15주"),
        ]
        assert result.netting_floor_price == pytest.approx(94.99)
        mirror = next(r for r in result.order_sheet if r["구분"] == "매도 (퉁치기)")
        assert (mirror["주문가"], mirror["수량"], mirror["변화율"]) == (pytest.approx(100.01), 15, "+1.0%")

    def test_fractional_quantities_are_kept(self):
        sheet = [
            {"구분": "매도 (익절)", "주문가": 98.0, "수량": 2.5, "변화율": "", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 4.25, "변화율": "", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, True)
        buy = next(r for r in result.order_sheet if r["구분"] == "매수")
        assert buy["수량"] == pytest.approx(1.75)
        mirror_buy = next(r for r in result.order_sheet if r["구분"] == "매수 (퉁치기)")
        assert mirror_buy["수량"] == pytest.approx(2.5)
