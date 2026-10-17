"""Broker costs (CapitalParams.fee_pct / sec_fee_pct), hand-computed.

Meritz US rates: 0.07% commission both ways, SEC fee 0.00206% on sells (at least $0.01).
With netting on, only the net shares are filled, so only they pay.
"""
from decimal import Decimal

import pandas as pd

from engines.dongpa_engine import CapitalParams, ModeParams, StrategyParams, buy_fee, run_backtest, sell_fee
from engines.order_book_engine import build_order_sheet, build_spread_orders
from tests.conftest import _make_price_df
from tests.test_order_book_engine import _DEFAULT_UI

FEE, SEC = 0.07, 0.00206
MODE = ModeParams(buy_cond_pct=3.0, tp_pct=2.0, max_hold_days=30, slices=4)  # budget 2500


def _run(closes, mode=MODE, fee=FEE, sec=SEC, **kwargs):
    target = _make_price_df(closes, start="2024-03-04")
    momo = _make_price_df([400.0] * 150, start="2023-10-02")
    params = StrategyParams(
        target_ticker="TEST", momentum_ticker="MOMO", defense=mode,
        offense=ModeParams(buy_cond_pct=50.0, tp_pct=50.0, max_hold_days=99, slices=1), **kwargs,
    )
    return run_backtest(target, momo, params, CapitalParams(initial_cash=10_000, fee_pct=fee, sec_fee_pct=sec))


def test_fee_helpers_round_to_the_cent_and_sec_fee_has_a_floor():
    assert buy_fee(Decimal("2472.00"), FEE) == Decimal("1.73")       # 1.7304
    assert sell_fee(Decimal("2521.44"), FEE, SEC) == Decimal("1.82")  # 1.77 + 0.05
    assert sell_fee(Decimal("105.06"), FEE, SEC) == Decimal("0.08")   # 0.07 + 0.01 minimum
    assert buy_fee(Decimal("100"), 0.0) == 0 and sell_fee(Decimal("0"), FEE, SEC) == 0


def test_gross_costs_without_netting():
    # day1 buy 24 @ 103 = 2472.00, fee 1.73 → cash 7526.27
    # day2 buy 23 @ 105.06 = 2416.38 (fee 1.69), sell 24 @ 105.06 = 2521.44 (fee 1.77 + SEC 0.05)
    result = _run([100.0, 103.0, 105.06], enable_netting=False)
    day1, day2 = result.journal.iloc[1], result.journal.iloc[2]
    assert day1["수수료"] == 1.73 and day1["현금"] == 7526.27
    assert day2["수수료"] == 3.51 and day2["현금"] == 7627.82
    first, second = result.trade_log.iloc[0], result.trade_log.iloc[1]
    assert first["매수수수료"] == 1.73 and first["매도수수료"] == 1.82
    assert first["실현손익"] == 45.89            # 2521.44 − 2472.00 − 1.73 − 1.82
    assert first["수익률(%)"] == 1.86            # on 2473.73 paid
    assert second["매수수수료"] == 1.69 and pd.isna(second["매도수수료"])
    assert day2["실현손익"] == 45.89


def test_netting_charges_only_the_net_fill():
    # day2 nets buy 23 against sell 24 → 1 share sold for 105.06: commission 0.07 + SEC minimum 0.01
    result = _run([100.0, 103.0, 105.06], enable_netting=True)
    day2 = result.journal.iloc[2]
    assert day2["수수료"] == 0.08
    assert day2["현금"] == 7631.25              # 7526.27 − 2416.38 + 2521.44 − 0.08
    first, second = result.trade_log.iloc[0], result.trade_log.iloc[1]
    assert first["매도수수료"] == 0.08 and first["실현손익"] == 47.63
    assert second["매수수수료"] == 0.0


def test_buy_needs_cash_for_the_commission_too():
    # one tranche of 10000 buys 100 @ 100 = 10000.00 exactly; + 7.00 commission does not fit
    whole = ModeParams(buy_cond_pct=0.0, tp_pct=2.0, max_hold_days=30, slices=1)
    assert _run([100.0, 100.0], whole).trade_log.empty
    assert len(_run([100.0, 100.0], whole, fee=0.0, sec=0.0).trade_log) == 1


def test_sell_fees_split_over_lots_sold_together_sum_to_the_day():
    # two lots expire on the same close: their sell fees add up to the day's sell cost
    mode = ModeParams(buy_cond_pct=50.0, tp_pct=90.0, max_hold_days=2, slices=4)
    result = _run([100.0, 100.0, 101.0, 101.0, 101.0], mode, enable_netting=False)
    sold = result.trade_log[result.trade_log["매도일자"] == result.trade_log["매도일자"].iloc[0]]
    day = result.journal[result.journal["거래일자"] == sold["매도일자"].iloc[0]].iloc[0]
    day_buy_fees = result.trade_log.loc[result.trade_log["매수일자"] == day["거래일자"], "매수수수료"].sum()
    assert round(sold["매도수수료"].sum() + day_buy_fees, 2) == day["수수료"]


def test_order_book_drops_a_buy_the_commission_makes_unaffordable():
    ui = {**_DEFAULT_UI, "defense_buy": 0.0, "cash_limited_buy": False, "fee_pct": FEE}
    sheet, ctx = build_order_sheet(pd.DataFrame(), 100.0, 10_000.0, 10_000.0, "defense", ui, False)
    assert not any(r["구분"] == "매수" for r in sheet) and ctx is None
    sheet, _ = build_order_sheet(pd.DataFrame(), 100.0, 10_000.0, 10_000.0, "defense", {**ui, "fee_pct": 0.0}, False)
    assert [r["수량"] for r in sheet if r["구분"] == "매수"] == [100]


def test_spread_steps_the_commission_makes_unaffordable_are_left_out():
    ui = {**_DEFAULT_UI, "spread_buy_levels": 3, "spread_buy_step": 1}
    ctx = {"buy_limit_price": 100.0, "effective_budget": 1000.0, "base_qty": 10, "cash": 1000.68,
           "tp_pct": 2.0, "sl_pct": 0.0}
    # step 1 fills 11 @ 90.91 = 1000.01: fits without fees, not with its 0.70 commission (1000.71);
    # steps 2 and 3 fill 12 @ 83.33 / 13 @ 76.92 = 999.96 + 0.70 = 1000.66, which still fits
    assert len(build_spread_orders({**ctx, "fee_pct": 0.0}, None, 100.0, ui, False)) == 3
    with_fee = build_spread_orders({**ctx, "fee_pct": FEE}, None, 100.0, ui, False)
    assert [r["주문가"] for r in with_fee] == [83.33, 76.92]
