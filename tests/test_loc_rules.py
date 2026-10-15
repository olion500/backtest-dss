"""Hand-computed LOC scenarios for run_backtest (CLAUDE.md "LOC Order Execution Rules").

The momentum ticker is flat, so the weekly RSI never signals and every session trades
the defense params. Each scenario is short enough to check by hand.
"""
import pandas as pd

from engines.dongpa_engine import CapitalParams, ModeParams, StrategyParams, run_backtest
from tests.conftest import _make_price_df


def _run(closes, mode: ModeParams, offense: ModeParams | None = None, **kwargs):
    target = _make_price_df(closes, start="2024-03-04")
    momo = _make_price_df([400.0] * 150, start="2023-10-02")
    params = StrategyParams(
        target_ticker="TEST",
        momentum_ticker="MOMO",
        defense=mode,
        offense=offense or ModeParams(buy_cond_pct=50.0, tp_pct=50.0, max_hold_days=99, slices=1),
        **kwargs,
    )
    return run_backtest(target, momo, params, CapitalParams(initial_cash=10_000))


def _day(result, i: int) -> pd.Series:
    return result.journal.iloc[i]


# budget = 10000 / 4 = 2500
MODE = ModeParams(buy_cond_pct=3.0, tp_pct=2.0, max_hold_days=30, slices=4)


def test_buy_fills_at_close_equal_to_limit_and_sells_at_tp_equal_to_close():
    # day1 limit = 100 × 1.03 = 103.00 = close → buy floor(2500 / 103) = 24 @ 103, TP 105.06
    # day2 close 105.06 ≥ TP → sell 24 @ 105.06 (limit 106.09 also buys floor(2500/105.06) = 23)
    result = _run([100.0, 103.0, 105.06], MODE, enable_netting=False)
    first = result.trade_log.iloc[0]
    assert first["매수주문가"] == 103.0
    assert first["매수체결가"] == 103.0
    assert first["매수수량"] == 24
    assert first["매수금액"] == 2472.0
    assert first["TP목표가"] == 105.06
    assert first["청산사유"] == "TP"
    assert first["매도평균"] == 105.06
    assert first["매도금액"] == 2521.44
    assert first["실현손익"] == 49.44
    assert first["수익률(%)"] == 2.0
    assert first["보유기간(일)"] == 2
    assert first["상태"] == "완료"
    assert first["매수모드"] == "안전"
    assert result.trade_log["거래ID"].tolist() == [1, 2]
    assert "_buy_timestamp" not in result.trade_log.columns
    assert _day(result, 1)["현금"] == 7528.0
    assert _day(result, 1)["예약요약"] == "매수 24주 @ 103.00 (예산 $2,472.00) | TP대기 24주 @ 105.06"
    assert _day(result, 2)["매도거래ID목록"] == "1"
    assert _day(result, 2)["누적수익률(%)"] == 0.49  # 49.44 / 10000
    # day2: buy 23 @ 105.06 (2416.38) and sell 24 @ 105.06 (2521.44)
    assert _day(result, 2)["현금"] == 7633.06


def test_no_buy_on_first_session_or_above_limit():
    # day0 never buys (no prev close); day1 103.01 > limit 103.00
    result = _run([100.0, 103.01], MODE)
    assert result.trade_log.empty
    assert pd.isna(_day(result, 0)["매수주문가"])
    assert _day(result, 1)["매수주문가"] == 103.0
    assert result.cash_end == 10_000
    assert result.equity.tolist() == [10_000, 10_000]


def test_lot_below_tp_keeps_holding_and_marks_to_close():
    # buy_cond 0: day1 limit 100 = close → buy 25 @ 100 (TP 102); day2 101.99 is above the
    # limit (no buy) and below TP (no sell).
    mode = ModeParams(buy_cond_pct=0.0, tp_pct=2.0, max_hold_days=30, slices=4)
    result = _run([100.0, 100.0, 101.99], mode)
    assert len(result.trade_log) == 1
    assert result.trade_log.iloc[0]["상태"] == "보유중"
    assert result.trade_log.iloc[0]["보유기간(일)"] == 2
    assert result.open_positions == 1
    day2 = _day(result, 2)
    assert day2["보유수량"] == 25
    assert day2["평가금액"] == 2549.75
    assert day2["Equity"] == 10_049.75
    assert day2["TP평균(보유)"] == 102.0
    assert result.equity.iloc[-1] == 10_049.75


def test_max_hold_forces_moc_on_the_last_allowed_session():
    # buy_cond −5%: day1 limit 95.00 = close → buy floor(2500 / 95) = 26; TP 104.50 never hit.
    # max_hold 3 → held days 1, 2, sold at the day-3 close (buy day counts as day 1).
    mode = ModeParams(buy_cond_pct=-5.0, tp_pct=10.0, max_hold_days=3, slices=4)
    result = _run([100.0, 95.0, 96.0, 97.0], mode)
    trade = result.trade_log.iloc[0]
    assert trade["매수수량"] == 26
    assert trade["청산사유"] == "MOC"
    assert trade["매도일자"] == "2024-03-07"
    assert trade["매도평균"] == 97.0
    assert trade["실현손익"] == 26 * 2.0
    assert trade["보유기간(일)"] == 3
    assert trade["수익률(%)"] == 2.11  # 52 / 2470
    assert _day(result, 2)["보유수량"] == 26
    assert pd.isna(_day(result, 2)["매수체결가"])  # no buy that day  # still held the day before


def test_stop_loss_sells_when_close_reaches_sl():
    # buy 26 @ 95, SL = 95 × 0.9 = 85.50; close 85.50 → SL sell at the close
    mode = ModeParams(buy_cond_pct=-5.0, tp_pct=10.0, max_hold_days=30, slices=4, stop_loss_pct=10.0)
    result = _run([100.0, 95.0, 85.5], mode)
    trade = result.trade_log.iloc[0]
    assert trade["SL목표가"] == 85.5
    assert trade["청산사유"] == "SL"
    assert trade["매도평균"] == 85.5
    assert trade["실현손익"] == -247.0  # 26 × (85.50 − 95)
    assert trade["수익률(%)"] == -10.0


def test_stop_loss_not_triggered_above_sl_and_disabled_at_zero():
    mode = ModeParams(buy_cond_pct=-5.0, tp_pct=10.0, max_hold_days=30, slices=4, stop_loss_pct=10.0)
    assert _run([100.0, 95.0, 85.51], mode).trade_log.iloc[0]["상태"] == "보유중"
    no_sl = ModeParams(buy_cond_pct=-5.0, tp_pct=10.0, max_hold_days=30, slices=4, stop_loss_pct=0.0)
    trade = _run([100.0, 95.0, 50.0], no_sl).trade_log.iloc[0]
    assert trade["SL목표가"] is None
    assert trade["상태"] == "보유중"


def test_tranche_base_resets_to_cash_after_a_sell():
    # day1 buy 25 @ 100 (cash 7500, TP 102); day2 close 106 > limit 103: TP sell 25 @ 106
    # → cash 10150, base 10150. day3 limit 109.18, close 104 → budget 2537.50, buy 24 @ 104.
    result = _run([100.0, 100.0, 106.0, 104.0], MODE)
    assert _day(result, 1)["일일트렌치예산"] == 2500.0
    assert _day(result, 1)["트렌치기반현금"] == 10_000.0
    assert _day(result, 2)["현금"] == 10_150.0
    assert _day(result, 2)["트렌치기반현금"] == 10_150.0
    assert _day(result, 2)["누적손익"] == 150.0
    assert _day(result, 2)["누적수익률(%)"] == 1.5
    day3 = _day(result, 3)
    assert day3["일일트렌치예산"] == 2537.5
    assert day3["매수수량"] == 24
    assert day3["현금"] == 7654.0


def test_budget_does_not_reset_without_a_sell():
    # two buys in a row: base stays at the initial cash, so both tranches are 2500
    mode = ModeParams(buy_cond_pct=3.0, tp_pct=10.0, max_hold_days=30, slices=4)
    result = _run([100.0, 100.0, 100.0], mode)
    assert _day(result, 2)["트렌치기반현금"] == 10_000.0
    assert _day(result, 2)["일일트렌치예산"] == 2500.0
    assert _day(result, 2)["현금"] == 5000.0


def test_skips_buy_when_fill_value_exceeds_cash():
    # slices 1: day1 buys floor(10000/100) = 100 @ 100 → cash 0; day2 cannot afford anything
    mode = ModeParams(buy_cond_pct=3.0, tp_pct=10.0, max_hold_days=30, slices=1)
    result = _run([100.0, 100.0, 99.0], mode)
    assert len(result.trade_log) == 1
    assert result.cash_end == 0.0


def test_cash_limited_buy_uses_remaining_cash():
    # 7 slices: budget 10000/7 = 1428.57; after 7 buys of 14 @ 100 (1400 each) the cash is 200
    # → legacy skips (14 × 100 > 200), the cash-limited buy spends min(1428.57, 200) on 2 shares.
    mode = ModeParams(buy_cond_pct=0.0, tp_pct=50.0, max_hold_days=30, slices=7)
    closes = [100.0] * 9
    legacy = _run(closes, mode)
    capped = _run(closes, mode, cash_limited_buy=True)
    assert legacy.trade_log["매수수량"].tolist() == [14] * 7
    assert capped.trade_log["매수수량"].tolist() == [14] * 7 + [2]
    assert capped.cash_end == 0.0


def test_netting_is_display_only():
    # day1 buy 25 @ 100 (TP 102). day2 close 102: buy floor(2500/102) = 24 @ 102 and TP-sell 25 @ 102.
    # Journal nets 24 → shows a net sell of 1 share; cash settles both legs.
    netted = _run([100.0, 100.0, 102.0], MODE)
    day2 = _day(netted, 2)
    assert day2["원매수수량"] == 24
    assert day2["원매도수량"] == 25
    assert day2["매수수량"] == 0
    assert day2["매수금액"] == 0.0
    assert day2["매도수량"] == 1
    assert day2["매도금액"] == 102.0
    assert day2["매도평균"] == 102.0
    assert day2["원매도평균"] == 102.0
    assert bool(day2["퉁치기적용"]) is True
    assert "순매도 1주" in day2["퉁치기상세"]
    assert day2["현금"] == 7602.0

    plain = _run([100.0, 100.0, 102.0], MODE, enable_netting=False)
    row = _day(plain, 2)
    assert (row["매수수량"], row["매도수량"]) == (24, 25)
    assert bool(row["퉁치기적용"]) is False
    assert row["현금"] == day2["현금"]


def test_drawdown_column_tracks_peak_equity():
    # buy 25 @ 100 then close 80.33: equity 7500 + 2008.25 = 9508.25 → DD −4.9175% → −4.92
    mode = ModeParams(buy_cond_pct=-50.0, tp_pct=50.0, max_hold_days=30, slices=4)
    result = _run([200.0, 100.0, 80.33], mode)
    assert _day(result, 2)["Equity"] == 9508.25
    assert _day(result, 2)["낙폭(DD%)"] == -4.92
    assert _day(result, 2)["등락률(%)"] == -19.67


def test_buy_limit_is_rounded_to_cents():
    # limit = 54.95 × 1.03 = 56.5985 → 56.60, so a 56.60 close fills
    result = _run([54.95, 56.6], MODE)
    assert _day(result, 1)["매수주문가"] == 56.6
    assert result.trade_log.iloc[0]["매수수량"] == 44  # floor(2500 / 56.60)


def test_budget_below_one_share_buys_nothing():
    result = _run([3000.0, 3000.0], MODE)  # 2500 budget < one share
    assert result.trade_log.empty
    assert _day(result, 1)["예약요약"] == "매수 없음 | TP대기 없음"


def test_small_stop_loss_pct():
    # SL 0.5%: 95 × 0.995 = 94.525 → 94.53 (half-up)
    mode = ModeParams(buy_cond_pct=-5.0, tp_pct=10.0, max_hold_days=30, slices=4, stop_loss_pct=0.5)
    result = _run([100.0, 95.0, 94.53], mode)
    assert result.trade_log.iloc[0]["SL목표가"] == 94.53
    assert result.trade_log.iloc[0]["청산사유"] == "SL"


def test_netting_net_buy_row():
    # max_hold 2: day1 buys 12 @ 200 (limit 206); day2 close 100 MOC-sells the 12 and buys 25.
    mode = ModeParams(buy_cond_pct=3.0, tp_pct=50.0, max_hold_days=2, slices=4)
    day2 = _day(_run([200.0, 200.0, 100.0], mode), 2)
    assert (day2["원매수수량"], day2["원매도수량"]) == (25, 12)
    assert day2["매수수량"] == 13
    assert day2["매수금액"] == 1300.0
    assert day2["매도수량"] == 0
    assert day2["매도금액"] == 0.0
    assert pd.isna(day2["매도평균"])
    assert day2["원매도평균"] == 100.0
    assert day2["퉁치기상세"] == "매수 25주, 매도 12주 → 순매수 13주"


def test_netting_net_sell_average():
    # slices 2 → budget 5000; buy_cond 10%. day1 55 @ 90 (TP 91.80); day2 close 99 = limit:
    # buy floor(5000/99) = 50 and TP-sell 55 → net sell 5 shares, 495 → average 99
    mode = ModeParams(buy_cond_pct=10.0, tp_pct=2.0, max_hold_days=30, slices=2)
    day2 = _day(_run([100.0, 90.0, 99.0], mode), 2)
    assert (day2["매도수량"], day2["매도금액"], day2["매도평균"]) == (5, 495.0, 99.0)
    assert day2["퉁치기상세"] == "매수 50주, 매도 55주 → 순매도 5주"


def test_full_offset_detail():
    # day1 12 @ 200 (max_hold 2); day2 close 206 = limit: buy floor(2500/206) = 12, MOC-sell 12
    mode = ModeParams(buy_cond_pct=3.0, tp_pct=50.0, max_hold_days=2, slices=4)
    day2 = _day(_run([200.0, 200.0, 206.0], mode), 2)
    assert (day2["매수수량"], day2["매도수량"]) == (0, 0)
    assert day2["퉁치기상세"] == "매수 12주, 매도 12주 → 상쇄"


def test_spread_ladder_hangs_from_the_netting_floor():
    # spread step 1, budget 2500.
    # day1 close 94, limit 103: base floor(2500/103) = 24, steps 2500/25 = 100.00, /26 = 96.15,
    #   /27 = 92.59 → close reaches two steps → 26 shares, TP 94 × 1.02 = 95.88.
    # day2 close 92.59, limit 96.82; the open lot's TP 95.88 ≤ limit → netting floor 95.87.
    #   base floor(2500/96.82) = 25. Netting: ref 95.87 → 26 shares, steps /27 = 92.59 → +1 = 26.
    #   No netting: ref 96.82 → 25 shares, steps /26 = 96.15, /27 = 92.59 → +2 = 27.
    closes = [100.0, 94.0, 92.59]
    netted = _run(closes, MODE, spread_buy_step=1)
    plain = _run(closes, MODE, spread_buy_step=1, enable_netting=False)
    assert netted.trade_log["매수수량"].tolist() == [26, 26]
    assert plain.trade_log["매수수량"].tolist() == [26, 27]
