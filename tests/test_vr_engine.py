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
