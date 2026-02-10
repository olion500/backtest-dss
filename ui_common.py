# -*- coding: utf-8 -*-
"""Shared UI utilities for Streamlit pages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------- Constants ----------------------

NAV_LINKS = [
    ("backtest.py", "backtest"),
    ("pages/2_orderBook.py", "orderBook"),
    ("pages/3_Optuna.py", "Optuna"),
]

SETTINGS_PATH = Path("config") / "order_book_settings.json"
CONFIG_DIR = Path("config")
LOOKBACK_DAYS = 400  # Extra days for RSI/MA warm-up

DEFAULT_PARAMS = {
    "target": "SOXL",
    "momentum": "QQQ",
    "bench": "SOXX",
    "log_scale": True,
    "mode_switch_strategy_index": 0,
    "ma_short": 3,
    "ma_long": 7,
    "rsi_high_threshold": 65.0,
    "rsi_mid_high": 60.0,
    "rsi_neutral": 50.0,
    "rsi_mid_low": 40.0,
    "rsi_low_threshold": 35.0,
    "enable_netting": True,
    "allow_fractional": False,
    "cash_limited_buy": False,
    "init_cash": 10000,
    "defense_slices": 7,
    "defense_buy": 3.0,
    "defense_tp": 0.2,
    "defense_sl": 0.0,
    "defense_hold": 30,
    "offense_slices": 7,
    "offense_buy": 5.0,
    "offense_tp": 2.5,
    "offense_sl": 0.0,
    "offense_hold": 7,
}


# ---------------------- Navigation ----------------------

def render_navigation() -> None:
    st.markdown(
        """
        <style>
        [data-testid='stSidebarNav'] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("### Pages")
    for path, label in NAV_LINKS:
        st.sidebar.page_link(path, label=label)
    st.sidebar.divider()


# ---------------------- Settings I/O ----------------------

def load_settings(config_path: Path | None = None) -> dict:
    """Load settings from a config JSON file."""
    path = config_path if config_path else SETTINGS_PATH
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_settings(payload: dict, config_path: Path | None = None) -> None:
    """Save settings to a config JSON file."""
    path = config_path if config_path else SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def get_available_config_files() -> list[Path]:
    """Get all JSON config files in the config directory."""
    if not CONFIG_DIR.exists():
        return []
    json_files = list(CONFIG_DIR.glob("*.json*"))
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


# ---------------------- Trade Metrics ----------------------

def compute_trade_metrics(
    trade_log: pd.DataFrame | None,
    initial_cash: float,
) -> dict[str, float | int | None] | None:
    """Compute realized trade metrics from trade log."""
    if trade_log is None or trade_log.empty:
        return None

    closed = trade_log[trade_log["상태"] == "완료"].copy()
    empty_result = {
        "trade_count": 0,
        "moc_count": 0,
        "net_profit": 0.0,
        "avg_hold_days": None,
        "avg_return_pct": None,
        "avg_gain_pct": None,
        "avg_loss_pct": None,
        "avg_gain": None,
        "avg_loss": None,
        "period_return_pct": None,
    }
    if closed.empty:
        return empty_result

    for col in ("실현손익", "보유기간(일)", "수익률(%)"):
        if col in closed.columns:
            closed[col] = pd.to_numeric(closed[col], errors="coerce")

    closed = closed.dropna(subset=["실현손익"])
    if closed.empty:
        return empty_result

    net_profit = float(closed["실현손익"].sum())
    trade_count = int(len(closed))
    moc_count = int((closed["청산사유"] == "MOC").sum()) if "청산사유" in closed.columns else 0
    avg_hold = float(closed["보유기간(일)"].mean()) if "보유기간(일)" in closed.columns else None
    avg_return_pct = None
    if "수익률(%)" in closed.columns and closed["수익률(%)"].notna().any():
        avg_return_pct = float(closed["수익률(%)"].dropna().mean())

    gain_series = closed.loc[closed["실현손익"] > 0, "실현손익"]
    loss_series = closed.loc[closed["실현손익"] < 0, "실현손익"]
    gain_pct_series = pd.Series(dtype=float)
    loss_pct_series = pd.Series(dtype=float)
    if "수익률(%)" in closed.columns:
        pct_series = pd.to_numeric(closed["수익률(%)"], errors="coerce")
        gain_pct_series = pct_series[pct_series > 0]
        loss_pct_series = pct_series[pct_series < 0]

    avg_gain = float(gain_series.mean()) if not gain_series.empty else None
    avg_loss = float(loss_series.mean()) if not loss_series.empty else None
    avg_gain_pct = float(gain_pct_series.mean()) if not gain_pct_series.empty else None
    avg_loss_pct = float(loss_pct_series.mean()) if not loss_pct_series.empty else None

    period_return_pct = None
    if initial_cash > 0:
        period_return_pct = (net_profit / initial_cash) * 100.0

    return {
        "trade_count": trade_count,
        "moc_count": moc_count,
        "net_profit": net_profit,
        "avg_hold_days": avg_hold,
        "avg_return_pct": avg_return_pct,
        "avg_gain_pct": avg_gain_pct,
        "avg_loss_pct": avg_loss_pct,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "period_return_pct": period_return_pct,
    }
