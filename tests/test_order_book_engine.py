"""Tests for order_book_engine pure functions."""

import pandas as pd
import pytest

from engines.order_book_engine import (
    BacktestState,
    NettingResult,
    apply_netting,
    build_holdings,
    build_order_sheet,
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
    def test_buy_only_no_positions(self):
        """No open positions, only a buy order is generated."""
        sheet, sl_sheet, ctx = build_order_sheet(
            pd.DataFrame(), 100.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        buy_rows = [r for r in sheet if r["구분"] == "매수"]
        assert len(buy_rows) == 1
        assert sl_sheet == []
        assert ctx is not None

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
        sheet, sl_sheet, ctx = build_order_sheet(
            trades, 98.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        tp_rows = [r for r in sheet if "TP" in r["구분"]]
        assert len(tp_rows) == 1

    def test_sell_sl_goes_to_sl_sheet(self):
        """SL sell order goes to the separate sl sheet."""
        trades = _make_open_trades([{
            "매수일자": "2024-01-02",
            "매수체결가": 100.0,
            "매수수량": 10,
            "TP목표가": 110.0,
            "SL목표가": 90.0,
            "최대보유일": 30,
            "보유기간(일)": 3,
        }])
        sheet, sl_sheet, _ = build_order_sheet(
            trades, 98.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        assert len(sl_sheet) == 1
        assert sl_sheet[0]["구분"] == "매도 (SL)"

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
        sheet, _, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        expiry_rows = [r for r in sheet if "만료" in r["구분"]]
        assert len(expiry_rows) == 1

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
        sheet, _, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        expiry_rows = [r for r in sheet if "만료" in r["구분"]]
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
        sheet, sl_sheet, _ = build_order_sheet(
            trades, 102.0, 5000.0, 1000.0, "defense", _DEFAULT_UI, False,
        )
        assert [r for r in sheet if "TP" in r["구분"]] == []
        assert sl_sheet == []
        expiry_rows = [r for r in sheet if "만료" in r["구분"]]
        assert len(expiry_rows) == 1
        assert expiry_rows[0]["수량"] == 10

    def test_no_cash_no_buy(self):
        """No cash → no buy order."""
        sheet, _, ctx = build_order_sheet(
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
            {"구분": "매도 (TP)", "주문가": 110.0, "수량": 10, "변화율": "+10%", "비고": ""},
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
            {"구분": "매도 (TP)", "주문가": 98.0, "수량": 10, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert "완전상쇄" in result.netting_msg
        # No plain 매수 or 매도 (TP) left — only scenario rows
        tp_rows = [r for r in result.order_sheet if r["구분"] == "매도 (TP)"]
        buy_plain = [r for r in result.order_sheet if r["구분"] == "매수" and "퉁치기" not in r.get("비고", "")]
        assert len(tp_rows) == 0

    def test_partial_netting_buy_larger(self):
        """Buy qty > sell qty → net buy remains."""
        sheet = [
            {"구분": "매도 (TP)", "주문가": 98.0, "수량": 5, "변화율": "+2%", "비고": ""},
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
            {"구분": "매도 (TP)", "주문가": 98.0, "수량": 15, "변화율": "+2%", "비고": ""},
            {"구분": "매수", "주문가": 100.0, "수량": 10, "변화율": "-3%", "비고": ""},
        ]
        result = apply_netting(sheet, 99.0, False)
        assert "순매도" in result.netting_msg

    def test_netting_produces_scenario_rows(self):
        """Netting should add floor (buy-only) and ceiling (sell-only) scenario rows."""
        sheet = [
            {"구분": "매도 (TP)", "주문가": 98.0, "수량": 10, "변화율": "+2%", "비고": ""},
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
        elif r["구분"].startswith("매도"):
            if close >= price - 1e-9:
                net -= qty
    return net


class TestNettingConsistency:
    """The netted sheet must reproduce the original net outcome at every
    close, and never list more sell shares than are actually held."""

    # Real-world case: expiring lot + two TP lots + buy 20 @ 145.97
    _SHEET = [
        {"구분": "매도 (만료)", "주문가": 135.47, "수량": 23, "변화율": "-", "비고": "잔여일: 1일"},
        {"구분": "매도 (TP)", "주문가": 144.28, "수량": 23, "변화율": "-", "비고": ""},
        {"구분": "매도 (TP)", "주문가": 137.18, "수량": 22, "변화율": "-", "비고": ""},
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
            {"구분": "매도 (TP)", "주문가": 95.0, "수량": 5, "변화율": "-", "비고": ""},
            {"구분": "매도 (TP)", "주문가": 98.0, "수량": 10, "변화율": "-", "비고": ""},
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
            if r["구분"] == "매수" and "미체결" in r.get("비고", "")
        )
        assert mirror_prices == [94.99, 97.99]

    def test_scenario_rows_are_informational(self):
        result = apply_netting(self._copy(), 135.47, False)
        assert result.scenario_rows
        # ranges: <135.47, 3 groups, >145.97 → 5 rows
        assert len(result.scenario_rows) == 5
        assert result.scenario_rows[0]["순결과"] == "순매수 20주"
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
        assert len(rows) == 3
        # Each row should be at progressively lower prices
        prices = [r["주문가"] for r in rows]
        assert prices == sorted(prices, reverse=True)

    def test_with_netting_floor(self):
        ctx = {
            "buy_limit_price": 100.0,
            "effective_budget": 1000.0,
            "base_qty": 10,
            "tp_pct": 2.0,
            "sl_pct": 0.0,
        }
        rows = build_spread_orders(ctx, 97.99, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 1}, False)
        # Uses netting_floor_price as reference, so prices differ from basic test
        assert len(rows) == 3
        for r in rows:
            assert r["주문가"] < 97.99

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
        assert "SL" in rows[0]["비고"]
