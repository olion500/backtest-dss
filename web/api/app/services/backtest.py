from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from engines.dongpa_engine import (
    BacktestResult,
    CapitalParams,
    Indicators,
    ModeParams,
    StrategyParams,
    compute_buy_and_hold_return,
    compute_equity_return,
    compute_indicators,
    compute_mode_bands,
    compute_trade_metrics,
    normalize_ohlcv,
    run_backtest,
    summarize,
)
from engines.order_book_engine import (
    apply_netting,
    build_holdings,
    build_order_sheet,
    build_spread_orders,
    extract_state,
)
from web.api.app.schemas import BacktestRequest, ModeSettings, StrategySettings
from web.api.app.serializers import dataframe_payload, json_value, records_payload
from web.api.app.services.market_data import MarketDataClient, market_data_client


LOOKBACK_DAYS = 1000
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class Execution:
    target_prices: pd.DataFrame
    momentum_prices: pd.DataFrame
    result: BacktestResult
    indicators: Indicators | None


PriceProvider = Callable[[str, date, date], pd.DataFrame]


def _mode_params(settings: ModeSettings) -> ModeParams:
    return ModeParams(
        buy_cond_pct=settings.buy_cond_pct,
        tp_pct=settings.tp_pct,
        max_hold_days=settings.max_hold_days,
        slices=settings.slices,
        stop_loss_pct=settings.stop_loss_pct or None,
    )


def build_engine_params(settings: StrategySettings, initial_cash: float) -> tuple[StrategyParams, CapitalParams]:
    params = StrategyParams(
        target_ticker=settings.target_ticker,
        momentum_ticker=settings.momentum_ticker,
        mode_switch_strategy=settings.mode_switch_strategy,
        enable_netting=settings.enable_netting,
        allow_fractional_shares=settings.allow_fractional_shares,
        cash_limited_buy=settings.cash_limited_buy,
        rsi_high_threshold=settings.rsi_high_threshold,
        rsi_mid_high=settings.rsi_mid_high,
        rsi_neutral=settings.rsi_neutral,
        rsi_mid_low=settings.rsi_mid_low,
        rsi_low_threshold=settings.rsi_low_threshold,
        ma_short_period=settings.ma_short_period,
        ma_long_period=settings.ma_long_period,
        roc_period=settings.roc_period,
        btc_lookback_days=settings.btc_lookback_days,
        btc_threshold_pct=settings.btc_threshold_pct,
        defense=_mode_params(settings.defense),
        offense=_mode_params(settings.offense),
    )
    return params, CapitalParams(initial_cash=initial_cash)


def _execute(
    request: BacktestRequest,
    provider: PriceProvider,
    include_indicators: bool,
) -> Execution:
    exclusive_end = request.end_date + timedelta(days=1)
    warmup_start = request.start_date - timedelta(days=LOOKBACK_DAYS)
    strategy = request.strategy

    target_raw = provider(strategy.target_ticker, request.start_date, exclusive_end)
    momentum_raw = provider(strategy.momentum_ticker, warmup_start, exclusive_end)
    btc_raw = None
    if strategy.mode_switch_strategy == "btc_overnight":
        btc_raw = provider(strategy.btc_ticker, warmup_start, exclusive_end)

    target = normalize_ohlcv(target_raw)
    target = target[
        (target.index >= pd.Timestamp(request.start_date))
        & (target.index <= pd.Timestamp(request.end_date))
    ]
    if target.empty:
        raise ValueError("선택한 기간에 대상 종목 데이터가 없습니다.")

    params, capital = build_engine_params(strategy, request.initial_cash)
    result = run_backtest(target, momentum_raw, params, capital, btc_data=btc_raw)
    indicators = None
    if include_indicators:
        _, indicators = compute_indicators(target, momentum_raw, params, btc_data=btc_raw)
    return Execution(target, normalize_ohlcv(momentum_raw), result, indicators)


def _equity_points(execution: Execution) -> list[dict]:
    price = execution.target_prices.get("Close", pd.Series(dtype=float))
    points = []
    for timestamp, equity in execution.result.equity.items():
        price_value = price.get(timestamp)
        points.append({
            "date": json_value(timestamp),
            "equity": json_value(equity),
            "price": json_value(price_value),
        })
    return points


def _backtest_payload(request: BacktestRequest, execution: Execution) -> dict:
    result = execution.result
    summary_metrics = summarize(result.equity)
    trade_metrics = compute_trade_metrics(result.trade_log, request.initial_cash)
    mode_bands = compute_mode_bands(result.journal)
    target_hold = compute_buy_and_hold_return(execution.target_prices)
    momentum_period = execution.momentum_prices[
        (execution.momentum_prices.index >= pd.Timestamp(request.start_date))
        & (execution.momentum_prices.index <= pd.Timestamp(request.end_date))
    ]

    return {
        "meta": {
            "target_ticker": request.strategy.target_ticker,
            "momentum_ticker": request.strategy.momentum_ticker,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "log_scale": request.log_scale,
        },
        "summary": {
            **json_value(summary_metrics),
            "Target Hold Return": json_value(target_hold),
            "Momentum Hold Return": json_value(compute_buy_and_hold_return(momentum_period)),
            "Strategy Return": json_value(compute_equity_return(result.equity)),
        },
        "realized_metrics": json_value(trade_metrics),
        "state": {
            "cash_end": json_value(result.cash_end),
            "open_positions": result.open_positions,
        },
        "equity": _equity_points(execution),
        "mode_bands": dataframe_payload(mode_bands),
        "journal": dataframe_payload(result.journal),
        "trade_log": dataframe_payload(result.trade_log),
    }


def run_backtest_view(
    request: BacktestRequest,
    provider: PriceProvider | None = None,
) -> dict:
    selected_provider = provider or market_data_client.download
    execution = _execute(request, selected_provider, include_indicators=False)
    return _backtest_payload(request, execution)


def _order_ui_values(request: BacktestRequest) -> dict:
    strategy = request.strategy
    return {
        "target": strategy.target_ticker,
        "momentum": strategy.momentum_ticker,
        "init_cash": request.initial_cash,
        "defense_slices": strategy.defense.slices,
        "defense_buy": strategy.defense.buy_cond_pct,
        "defense_tp": strategy.defense.tp_pct,
        "defense_sl": strategy.defense.stop_loss_pct,
        "defense_hold": strategy.defense.max_hold_days,
        "offense_slices": strategy.offense.slices,
        "offense_buy": strategy.offense.buy_cond_pct,
        "offense_tp": strategy.offense.tp_pct,
        "offense_sl": strategy.offense.stop_loss_pct,
        "offense_hold": strategy.offense.max_hold_days,
        "spread_buy_levels": request.spread_buy_levels,
        "spread_buy_step": request.spread_buy_step,
    }


def run_order_book_view(
    request: BacktestRequest,
    provider: PriceProvider | None = None,
) -> dict:
    selected_provider = provider or market_data_client.download
    execution = _execute(request, selected_provider, include_indicators=True)
    result = execution.result
    if result.journal.empty or execution.indicators is None:
        raise ValueError("주문 시트를 만들 거래 기록이 없습니다.")

    ui_values = _order_ui_values(request)
    state = extract_state(result.journal, execution.indicators, ui_values, request.initial_cash)
    if result.trade_log.empty or "상태" not in result.trade_log:
        open_trades = pd.DataFrame()
    else:
        open_trades = result.trade_log[result.trade_log["상태"] != "완료"].copy()

    holdings = build_holdings(open_trades, state.prev_close)
    order_sheet, stop_loss_sheet, spread_context = build_order_sheet(
        open_trades,
        state.prev_close,
        state.current_cash,
        state.tranche_budget,
        state.current_mode,
        ui_values,
        request.strategy.allow_fractional_shares,
    )

    netting_message = ""
    netting_details: list[dict] = []
    scenario_rows: list[dict] = []
    netting_floor = None
    if request.strategy.enable_netting:
        netting = apply_netting(
            order_sheet,
            state.prev_close,
            request.strategy.allow_fractional_shares,
        )
        order_sheet = netting.order_sheet
        netting_message = netting.netting_msg
        netting_details = netting.netting_details
        scenario_rows = netting.scenario_rows
        netting_floor = netting.netting_floor_price

    if spread_context is not None:
        order_sheet.extend(build_spread_orders(
            spread_context,
            netting_floor,
            state.prev_close,
            ui_values,
            request.strategy.allow_fractional_shares,
        ))
    order_sheet.sort(key=lambda row: float(row.get("주문가", 0)), reverse=True)

    payload = _backtest_payload(request, execution)
    payload["order_book"] = {
        "state": json_value({
            "last_date": state.last_date,
            "current_mode": state.current_mode,
            "current_cash": state.current_cash,
            "current_position_qty": state.current_position_qty,
            "prev_close": state.prev_close,
            "tranche_budget": state.tranche_budget,
            "rsi_value": state.rsi_value,
        }),
        "holdings": records_payload(holdings),
        "orders": records_payload(order_sheet),
        "stop_loss_orders": records_payload(stop_loss_sheet),
        "netting_message": netting_message,
        "netting_details": records_payload(netting_details),
        "netting_scenarios": records_payload(scenario_rows),
    }
    return payload


def _load_config(filename: str) -> dict:
    path = REPO_ROOT / "config" / filename
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def default_request() -> BacktestRequest:
    today = date.today()
    raw = _load_config("strategy.json")
    personal = _load_config("personal_settings.json")

    start_date = date(today.year - 4, 1, 1)
    try:
        start_date = date.fromisoformat(personal["start_date"])
    except (KeyError, TypeError, ValueError):
        pass

    strategy = StrategySettings(
        target_ticker=raw.get("target", "SOXL"),
        momentum_ticker=raw.get("momentum", "QQQ"),
        enable_netting=personal.get("enable_netting", True),
        allow_fractional_shares=personal.get("allow_fractional", False),
        cash_limited_buy=raw.get("cash_limited_buy", False),
        rsi_high_threshold=raw.get("rsi_high_threshold", 65),
        rsi_mid_high=raw.get("rsi_mid_high", 60),
        rsi_neutral=raw.get("rsi_neutral", 50),
        rsi_mid_low=raw.get("rsi_mid_low", 40),
        rsi_low_threshold=raw.get("rsi_low_threshold", 35),
        defense=ModeSettings(
            slices=raw.get("defense_slices", 7),
            buy_cond_pct=raw.get("defense_buy", 3),
            tp_pct=raw.get("defense_tp", 0.2),
            stop_loss_pct=raw.get("defense_sl", 0),
            max_hold_days=raw.get("defense_hold", 30),
        ),
        offense=ModeSettings(
            slices=raw.get("offense_slices", 7),
            buy_cond_pct=raw.get("offense_buy", 5),
            tp_pct=raw.get("offense_tp", 2.5),
            stop_loss_pct=raw.get("offense_sl", 0),
            max_hold_days=raw.get("offense_hold", 7),
        ),
    )
    return BacktestRequest(
        start_date=start_date,
        end_date=today,
        initial_cash=float(personal.get("init_cash", 10_000)),
        log_scale=personal.get("log_scale", True),
        spread_buy_levels=int(personal.get("spread_buy_levels", 5)),
        spread_buy_step=int(personal.get("spread_buy_step", 1)),
        strategy=strategy,
    )
