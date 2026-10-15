"""Backtest fills through the order book's step (spread) buy ladder."""
from decimal import Decimal

from engines.dongpa_engine import (
    CapitalParams,
    ModeParams,
    StrategyParams,
    netting_floor_price,
    run_backtest,
    spread_fill_qty,
    spread_ladder,
)
from engines.order_book_engine import build_spread_orders
from tests.conftest import _make_price_df


MODE = ModeParams(buy_cond_pct=3.0, tp_pct=50.0, max_hold_days=30, slices=7)


def _params(**kwargs) -> StrategyParams:
    return StrategyParams(
        target_ticker="TEST",
        momentum_ticker="MOMO",
        defense=MODE,
        offense=MODE,
        **kwargs,
    )


def _first_buy_qty(close_day2: float, **kwargs) -> float:
    target = _make_price_df([100.0, close_day2])
    momo = _make_price_df([400.0 + i for i in range(120)], start="2023-09-01")
    result = run_backtest(target, momo, _params(**kwargs), CapitalParams(initial_cash=10_000))
    return float(result.trade_log.iloc[0]["매수수량"])


class TestSpreadLadder:
    def test_prices_match_budget_share_counts(self):
        # budget 1000 @ ref 100 → 10 shares; step n buys 10 + 3n shares with the budget
        ladder = spread_ladder(1000, 100, levels=3, step=3)
        assert ladder == [
            (Decimal("76.92"), Decimal("3")),
            (Decimal("62.50"), Decimal("3")),
            (Decimal("52.63"), Decimal("3")),
        ]

    def test_stops_below_half_of_reference(self):
        ladder = spread_ladder(1000, 100, levels=20, step=3)
        assert all(price >= 50 for price, _ in ladder)
        assert len(ladder) == 3

    def test_order_book_rows_use_same_prices(self):
        ctx = {"buy_limit_price": 100.0, "effective_budget": 1000.0, "base_qty": 10, "tp_pct": 2.0, "sl_pct": 0.0}
        rows = build_spread_orders(ctx, None, 100.0, {"spread_buy_levels": 3, "spread_buy_step": 3}, False)
        assert [r["주문가"] for r in rows] == [76.92, 62.5, 52.63]
        assert [r["수량"] for r in rows] == [3, 3, 3]


class TestBacktestStepFill:
    # budget = 10000 / 7 = 1428.57, limit = 103.00 → base 13 shares.
    # Steps (step 3): 1428.57/16 = 89.29, /19 = 75.19, /22 = 64.94, /25 = 57.14, ...

    def test_legacy_buys_whole_budget_at_close(self):
        assert _first_buy_qty(80.0) == 17  # floor(1428.57 / 80)

    def test_ladder_buys_only_reached_steps(self):
        assert _first_buy_qty(80.0, spread_buy_step=3) == 16  # 13 + 3

    def test_ladder_close_above_first_step_buys_base_only(self):
        assert _first_buy_qty(95.0, spread_buy_step=3) == 13

    def test_ladder_has_no_level_cap(self):
        # close 50 reaches steps 1-5 (89.29 … 50.79)
        assert _first_buy_qty(50.0, spread_buy_step=3) == 13 + 5 * 3

    def test_step_one_matches_legacy(self):
        assert _first_buy_qty(80.0, spread_buy_step=1) == _first_buy_qty(80.0)


class TestSpreadFillQty:
    def test_counts_reached_steps(self):
        # budget 1000 @ ref 100 → steps 76.92, 62.50, 52.63, 45.45 …
        assert spread_fill_qty(1000, 100, 77, step=3) == 0
        assert spread_fill_qty(1000, 100, 76.92, step=3) == 3
        assert spread_fill_qty(1000, 100, 50, step=3) == 9


class TestLadderGuards:
    def test_no_ladder_for_non_positive_inputs(self):
        assert spread_ladder(0, 100, levels=3, step=1) == []
        assert spread_ladder(1000, 0, levels=3, step=1) == []
        assert spread_ladder(1000, 100, levels=0, step=1) == []
        assert spread_ladder(1000, 100, levels=3, step=0) == []
        assert spread_fill_qty(0, 100, 50, step=1) == 0
        assert spread_fill_qty(1000, 0, 50, step=1) == 0
        assert spread_fill_qty(1000, 100, 0, step=1) == 0
        assert spread_fill_qty(1000, 100, 50, step=0) == 0

    def test_no_ladder_when_budget_buys_no_share(self):
        assert spread_ladder(99, 100, levels=3, step=1) == []
        assert spread_fill_qty(99, 100, 10, step=1) == 0

    def test_a_step_exactly_half_the_reference_is_kept(self):
        # budget 1000 @ 100 → 10 shares; step 10 → 1000/20 = 50.00 = −50%, still allowed
        assert spread_ladder(1000, 100, levels=3, step=10) == [(Decimal("50.00"), Decimal("10"))]

    def test_fill_qty_has_no_level_cap_but_stops_at_close(self):
        # step 1: 1000/11 = 90.91 … 1000/20 = 50.00 → close 50 reaches 10 steps
        assert spread_fill_qty(1000, 100, 50, step=1) == 10
        assert spread_fill_qty(1000, 100, 90.91, step=1) == 1
        assert spread_fill_qty(1000, 100, 90.92, step=1) == 0


class TestNettingFloorPrice:
    @staticmethod
    def _lot(tp, days=0, max_hold=30):
        return {"tp": Decimal(str(tp)), "days": days, "max_hold": max_hold}

    def test_cheapest_tp_at_or_below_limit_minus_a_cent(self):
        lots = [self._lot(101.0), self._lot(99.5), self._lot(104.0)]
        assert netting_floor_price(lots, Decimal("101.00")) == Decimal("99.49")

    def test_ignores_tps_above_the_limit(self):
        assert netting_floor_price([self._lot(101.01)], Decimal("101.00")) is None
        assert netting_floor_price([], Decimal("101.00")) is None

    def test_ignores_lots_expiring_today(self):
        # days 6 + 1 ≥ 7 → MOC sell today, no mirror buy; days 5 + 1 < 7 → counts
        assert netting_floor_price([self._lot(99.0, days=6, max_hold=7)], Decimal("101")) is None
        assert netting_floor_price([self._lot(99.0, days=5, max_hold=7)], Decimal("101")) == Decimal("98.99")
        # max_hold 0 means no expiry
        assert netting_floor_price([self._lot(99.0, days=50, max_hold=0)], Decimal("101")) == Decimal("98.99")
