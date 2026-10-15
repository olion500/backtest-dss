from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from engines.vr_engine import VRConfig, build_orders, next_v, run_vr
from web.api.app.services.vr import VRAccount, load_vr_account_models, run_vr_view


def _bars(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows: (date, open, high, low, close)"""
    frame = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame.set_index("Date")


def _config(**overrides) -> VRConfig:
    values = dict(start_date=date(2026, 10, 3), start_shares=800, start_pool=15112.35, start_v=67915.68)
    values.update(overrides)
    return VRConfig(**values)


def test_ladder_matches_the_sheet():
    # 구글 시트 VR 대시보드 (2026-10-03 사이클, 밴드 -20% / +25%)
    v = 67915.68
    buys = [o for o in build_orders(800, v, 15112.35, 81.28, _config(band_pct=20)) if o.side == "buy"]
    sells = [o for o in build_orders(800, v, 15112.35, 81.28, _config(band_pct=25)) if o.side == "sell"]
    assert [(o.price, o.qty) for o in buys[:4]] == [(67.33, 8), (66.67, 8), (66.02, 8), (65.38, 8)]
    assert [(o.price, o.qty) for o in sells[:3]] == [(106.12, 1), (106.25, 1), (106.38, 1)]


def test_buy_ladder_stays_within_pool_limit():
    config = _config()
    v_min = 67915.68 * 0.85
    buys = [o for o in build_orders(800, 67915.68, 15112.35, 81.28, config) if o.side == "buy"]
    shares = sum(o.qty for o in buys)
    cost = sum(v_min / (800 + level) for level in range(shares))
    assert cost <= 15112.35 * 0.75
    assert cost + v_min / (800 + shares) > 15112.35 * 0.75


def test_sell_ladder_stops_at_cap():
    sells = [o for o in build_orders(800, 67915.68, 15112.35, 81.28, _config()) if o.side == "sell"]
    assert sells and all(o.price <= 81.28 * 1.4 for o in sells)


def test_next_v_formula():
    # 실력공식: V + P/G + (E - V) / (2√G) + 적립
    assert next_v(1000, 200, 900, 10, True, 50) == pytest.approx(1000 + 20 + (-100) / (2 * math.sqrt(10)) + 50)
    assert next_v(1000, 200, 900, 10, False, 50) == pytest.approx(1070)


def test_cycles_fill_and_roll_over():
    bars = _bars([
        ("2026-10-02", 81, 82, 80, 81.28),
        ("2026-10-05", 81, 82, 50, 52),        # buy ladder hit
        ("2026-10-16", 52, 53, 51, 52),
        ("2026-10-19", 52, 53, 51, 52),        # next cycle
    ])
    cycles = run_vr(bars, _config(deposit=500), date(2026, 10, 19))
    assert len(cycles) == 2
    first, second = cycles
    filled = [o for o in first.orders if o.filled_on]
    assert filled and all(o.side == "buy" for o in filled)
    assert first.shares == 800 + sum(o.qty for o in filled)
    expected_v = next_v(67915.68, first.pool, first.shares * 52, 10, True, 500)
    assert second.v == pytest.approx(round(expected_v, 2))
    assert second.pool_start == pytest.approx(first.pool + 500)
    assert second.principal == pytest.approx(first.principal + 500)


def test_adjustment_overrides_cycle_start():
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-10-19", 81, 82, 80, 81)])
    config = _config(adjustments=[{"date": date(2026, 10, 17), "shares": 810, "pool": 14000.0}])
    second = run_vr(bars, config, date(2026, 10, 19))[1]
    assert second.shares_start == 810
    assert second.pool_start == pytest.approx(14000.0)


def test_g_steps_every_n_cycles():
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28)])
    cycles = run_vr(bars, _config(g_step_cycles=2), date(2026, 12, 1))
    assert [c.g for c in cycles] == [10, 10, 11, 11, 12]


def test_vr_view_and_account_files(tmp_path):
    (tmp_path / "main.json").write_text(
        '{"name": "VR", "start_date": "2026-10-03", "start_shares": 800, "start_pool": 15112.35,'
        ' "start_v": 67915.68, "deposit": 500, "deposit_changes": [{"from": "2026-10-31", "deposit": 600}]}',
        encoding="utf-8",
    )
    (tmp_path / "broken.json").write_text('{"name": "x"}', encoding="utf-8")
    accounts = load_vr_account_models(tmp_path)
    assert [a.id for a in accounts] == ["main"]
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-11-02", 81, 82, 80, 81)])
    view = run_vr_view(accounts[0], provider=lambda *_: bars, today=date(2026, 11, 2))
    assert view["current"]["index"] == 2 and view["current"]["deposit"] == 600
    assert view["orders"] and {"side", "price", "qty"} <= set(view["orders"][0])
    assert view["account"]["deposit_changes"][0]["from"] == "2026-10-31"


def test_vr_view_before_start_shows_first_orders():
    account = VRAccount(id="a", name="a", start_date=date(2026, 10, 17), start_shares=10, start_pool=1000)
    bars = _bars([("2026-10-09", 81, 82, 80, 81.0)])
    view = run_vr_view(account, provider=lambda *_: bars, today=date(2026, 10, 12))
    assert view["current"]["index"] == 0 and view["current"]["v"] == pytest.approx(810.0)
    assert view["orders"]
    assert view["outlook"]["day"] == 1  # before the start: clamped to the first day
    assert view["today"] == "2026-10-12"


def test_vr_view_outlook_and_daily():
    account = VRAccount(id="a", name="a", start_date=date(2026, 10, 3), start_shares=800, start_pool=15112.35,
                        start_v=67915.68, deposit=500)
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-10-05", 81, 82, 70, 72), ("2026-10-06", 72, 73, 71, 72.5)])
    view = run_vr_view(account, provider=lambda *_: bars, today=date(2026, 10, 8))
    out = view["outlook"]
    filled = [o for o in view["orders"] if o["filled_on"]]
    spent = sum(o["qty"] * o["fill_price"] for o in filled)
    assert out["day"] == 6 and out["cycle_days"] == 14
    assert out["buy_budget_left"] == pytest.approx(15112.35 * 0.75 - spent, abs=0.01)
    assert out["buy_line"] == max(o["price"] for o in view["orders"] if o["side"] == "buy" and not o["filled_on"])
    current = view["current"]
    expected = next_v(67915.68, current["pool"], current["shares"] * 72.5, 10, True, 500)
    assert out["next_v"] == pytest.approx(round(expected, 2)) and out["next_deposit"] == 500
    assert [d["date"] for d in view["daily"]] == ["2026-10-05", "2026-10-06"]
    assert view["daily"][-1]["stock_value"] == pytest.approx(current["shares"] * 72.5)
    # low 70 reaches three 8-share chunks: V최소/807 = 71.53, /815 = 70.83, /823 = 70.14
    assert sorted(o["fill_price"] for o in filled) == [70.14, 70.83, 71.53]
    assert (current["bought"], current["sold"], current["shares"]) == (24, 0, 824)
    assert current["pool"] == pytest.approx(15112.35 - 1700.0)
    assert current["stock_value"] == 824 * 72.5
    assert current["total"] == pytest.approx(824 * 72.5 + 13412.35)
    assert current["principal"] == pytest.approx(800 * 81.28 + 15112.35)
    assert current["return_pct"] == -8.72  # 73152.35 / 80136.35 − 1
    assert (current["v_min"], current["v_max"]) == (57728.33, 78103.03)
    assert out["sell_line"] == 97.63
    assert out["buy_budget"] == 11334.26
    assert out["next_start"] == "2026-10-17" and out["next_g"] == 10


def _flat_bars(start: str, days: int, price: float = 80.0) -> pd.DataFrame:
    dates = pd.bdate_range(start=start, periods=days)
    return _bars([(d.strftime("%Y-%m-%d"), price, price, price, price) for d in dates])


def test_origin_date_sets_weeks_and_g_steps():
    # 시트 시작일 2023-07-01 → 2026-10-03은 85사이클(170주) 뒤: G 10에서 26사이클마다 +1 → 13
    config = _config(origin_date=date(2023, 7, 1), deposit=500)
    cycles = run_vr(_flat_bars("2026-09-28", 300), config, date(2027, 7, 1))
    assert cycles[0].week == 170 and cycles[1].week == 172
    assert cycles[0].g == 13
    # the 104th cycle from the VR start (19 after this config's first) is the next step
    assert cycles[18].g == 13 and cycles[19].g == 14
    assert cycles[19].start == date(2027, 6, 26)


def test_without_origin_counts_from_start_date():
    cycles = run_vr(_flat_bars("2026-09-28", 30), _config(), date(2026, 10, 20))
    assert cycles[0].week == 0 and cycles[0].g == 10


def test_origin_must_be_whole_cycles_before_start():
    with pytest.raises(ValueError, match="2주 단위"):
        run_vr(_flat_bars("2026-09-28", 10), _config(origin_date=date(2023, 7, 2)), date(2026, 10, 5))


def test_account_converts_legacy_start_week_to_origin():
    account = VRAccount.model_validate({
        "id": "main", "name": "VR", "start_week": 170, "start_date": "2026-10-03",
        "start_shares": 800, "start_pool": 15112.35,
    })
    assert account.origin_date == date(2023, 7, 1)
    assert account.engine_config().origin_date == date(2023, 7, 1)


def test_sell_ladder_runs_up_to_the_cap():
    # V최대 = 67915.68 × 1.15 = 78103.03; cap = 81.28 × 1.4 = 113.79
    # one share per level at V최대/800, V최대/799, … while ≤ cap → down to Q = 687 (114 levels)
    sells = [o for o in build_orders(800, 67915.68, 15112.35, 81.28, _config()) if o.side == "sell"]
    assert len(sells) == 114
    assert (sells[0].price, sells[-1].price) == (97.63, 113.69)
    assert all(o.qty == 1 for o in sells)


def test_buy_chunk_priced_at_its_lowest_level():
    # V최소 = 67915.68 × 0.85 = 57728.33; chunk 1 = levels 800…807 → price V최소/807
    buys = [o for o in build_orders(800, 67915.68, 15112.35, 81.28, _config()) if o.side == "buy"]
    assert (buys[0].price, buys[0].qty) == (71.53, 8)
    assert buys[1].price == 70.83  # V최소/815


def _one_bar_cycle(bar: tuple, **overrides):
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-10-05", *bar)])
    return run_vr(bars, _config(**overrides), date(2026, 10, 5))[0]


def test_limit_buy_fills_at_its_price_when_low_reaches_it():
    cycle = _one_bar_cycle((75, 76, 71.5, 72))
    filled = [o for o in cycle.orders if o.filled_on]
    assert [(o.side, o.price, o.qty, o.fill_price) for o in filled] == [("buy", 71.53, 8, 71.53)]
    assert cycle.shares == 808
    assert cycle.pool == pytest.approx(15112.35 - 8 * 71.53)


def test_limit_buy_fills_at_the_open_on_a_gap_down():
    cycle = _one_bar_cycle((60, 61, 59, 60))
    first = cycle.orders[0]
    assert first.side == "buy" and first.fill_price == 60.0


def test_limit_sell_fills_at_the_open_on_a_gap_up():
    # levels V최대/800 … V최대/781 (100.0039 → order price 100.00) are ≤ 100:
    # 20 one-share sells, all at the 100 open
    cycle = _one_bar_cycle((100, 100, 99, 99.5))
    sold = [o for o in cycle.orders if o.filled_on]
    assert all(o.side == "sell" and o.fill_price == 100.0 for o in sold)
    assert len(sold) == 20
    assert cycle.shares == 780
    assert cycle.pool == pytest.approx(15112.35 + 2000)


def test_loc_orders_fill_only_at_the_close():
    # the low (70) crosses two buy chunks but the close (71.60) is above both → no LOC fill
    assert not any(o.filled_on for o in _one_bar_cycle((75, 76, 70, 71.6), order_type="loc").orders)
    cycle = _one_bar_cycle((75, 76, 70, 71.0), order_type="loc")
    assert [(o.price, o.fill_price) for o in cycle.orders if o.filled_on] == [(71.53, 71.0)]


def test_first_cycle_defaults():
    cycle = _one_bar_cycle((81, 82, 80, 81))
    assert cycle.deposit == 0.0
    assert cycle.principal == pytest.approx(800 * 81.28 + 15112.35)
    assert cycle.v_min == pytest.approx(67915.68 * 0.85)
    assert cycle.v_max == pytest.approx(67915.68 * 1.15)
    assert (cycle.start, cycle.end) == (date(2026, 10, 3), date(2026, 10, 16))
    # no fill (low 80 > 71.53, high 82 < 97.63): 800 × 81 + pool
    assert cycle.stock_value == 64800.0
    assert cycle.total == pytest.approx(64800.0 + 15112.35)


def test_order_at_the_exact_price_fills():
    assert _one_bar_cycle((75, 76, 71.53, 72)).orders[0].fill_price == 71.53
    assert _one_bar_cycle((75, 76, 70, 71.53), order_type="loc").orders[0].fill_price == 71.53
    sell = next(o for o in _one_bar_cycle((95, 97.63, 94, 95)).orders if o.side == "sell")
    assert (sell.price, sell.fill_price) == (97.63, 97.63)
    sell = next(o for o in _one_bar_cycle((95, 99, 94, 97.63), order_type="loc").orders if o.side == "sell")
    assert sell.fill_price == 97.63


def test_next_v_never_negative():
    assert next_v(1000, 0, 1000, 10, False, -2000) == 0.0


def test_no_ladder_without_shares():
    assert build_orders(0, 1000, 500, 80, _config()) == []


def test_sell_ladder_down_to_the_last_share_and_partial_chunks():
    # V 301, band 15 → V최소 255.85, V최대 346.15. Pool 250 → buy budget 187.50:
    # 255.85/3 = 85.28 + 255.85/4 = 63.96 = 149.25 ≤ 187.5, + 255.85/5 = 51.17 → over.
    orders = build_orders(3, 301, 250, 100, _config(sell_cap_pct=1000, sell_unit=2))
    buys = [(o.price, o.qty) for o in orders if o.side == "buy"]
    sells = [(o.price, o.qty) for o in orders if o.side == "sell"]
    assert buys == [(63.96, 2)]
    # sells at 346.15/3, /2, /1 (cap 1100): first chunk of two priced at its top level, then one
    assert sells[0][1] == 2 and sells[0][0] == pytest.approx(173.075, abs=0.006)
    assert sells[1] == (346.15, 1)


def test_g_steps_disabled_and_fractional():
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28)])
    assert [c.g for c in run_vr(bars, _config(g_step_cycles=0), date(2026, 11, 20))] == [10, 10, 10, 10]
    assert [c.g for c in run_vr(bars, _config(g_step_cycles=1, g_step=0.5), date(2026, 11, 1))] == [10, 10.5, 11]


def test_next_cycle_v_uses_the_previous_cycles_g():
    # G steps every cycle (10 → 11) but V for cycle 2 grows with cycle 1's G = 10
    cycles = run_vr(_flat_bars("2026-09-28", 20), _config(g_step_cycles=1), date(2026, 10, 17))
    e, p, v = 800 * 80.0, 15112.35, 67915.68
    expected = v + p / 10 + (e - v) / (2 * math.sqrt(10))
    assert cycles[1].g == 11
    assert cycles[1].v == pytest.approx(round(expected, 2))


def test_adjustment_on_the_cycle_end_applies_to_that_cycle():
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-10-19", 81, 82, 80, 81)])
    config = _config(adjustments=[{"date": date(2026, 10, 30), "shares": 900}])
    cycles = run_vr(bars, config, date(2026, 10, 19))
    assert cycles[0].shares_start == 800
    assert cycles[1].shares_start == 900


def test_cycle_window_includes_its_first_and_last_day():
    # start Monday 10-05: the 10-05 bar is the first session, not the reference close
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28), ("2026-10-05", 81, 82, 80, 90), ("2026-10-18", 60, 61, 50, 55)])
    cycles = run_vr(bars, _config(start_date=date(2026, 10, 5), start_v=None), date(2026, 10, 19))
    assert cycles[0].v == 800 * 81.28
    assert cycles[0].days[0]["date"] == date(2026, 10, 5)
    assert cycles[0].end == date(2026, 10, 18)
    assert cycles[0].days[-1]["date"] == date(2026, 10, 18)
    assert any(o.filled_on == date(2026, 10, 18) for o in cycles[0].orders)


def test_cycle_without_a_session_uses_the_latest_close():
    bars = _bars([("2026-10-02", 81, 82, 80, 81.28)])
    cycle = run_vr(bars, _config(), date(2026, 10, 3))[0]
    assert (cycle.close, cycle.close_date) == (81.28, date(2026, 10, 2))
    assert cycle.days == []
    assert cycle.stock_value == pytest.approx(800 * 81.28)


def test_holding_cost_is_a_moving_average():
    from engines.vr_engine import Order
    from web.api.app.services.vr import _holding_cost

    def cycle(shares_start, orders):
        c = run_vr.__globals__["Cycle"](index=0, week=0, start=date(2026, 1, 3), end=date(2026, 1, 16), g=10, v=1,
                                        v_min=1, v_max=1, deposit=0, principal=0, shares_start=shares_start,
                                        pool_start=0, orders=orders)
        return c

    account = VRAccount(id="a", name="a", start_date=date(2026, 1, 3), start_shares=10, start_pool=100, principal=1100)
    cycles = [
        cycle(10, [Order("buy", 50.0, 10, date(2026, 1, 5), 50.0)]),          # 10 @ 100 + 10 @ 50 → avg 75
        cycle(20, [Order("sell", 90.0, 4, date(2026, 1, 20), 90.0)]),         # selling keeps avg 75 → 16 × 75
        cycle(32, []),                                                          # adjustment doubles holding, avg kept
    ]
    assert _holding_cost(account, cycles[:1]) == pytest.approx(1500.0)
    assert _holding_cost(account, cycles[:2]) == pytest.approx(16 * 75.0)
    assert _holding_cost(account, cycles) == pytest.approx(32 * 75.0)
