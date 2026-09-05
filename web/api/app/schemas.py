from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


TICKER_PATTERN = re.compile(r"^[A-Za-z0-9.^=-]{1,20}$")


class ModeSettings(BaseModel):
    slices: int = Field(ge=1, le=500)
    buy_cond_pct: float = Field(ge=-100, le=100)
    tp_pct: float = Field(ge=0, le=1000)
    stop_loss_pct: float = Field(default=0, ge=0, le=100)
    max_hold_days: int = Field(ge=1, le=3650)


class StrategySettings(BaseModel):
    target_ticker: str = "SOXL"
    momentum_ticker: str = "QQQ"
    mode_switch_strategy: Literal["rsi", "ma_cross", "roc", "btc_overnight"] = "rsi"
    enable_netting: bool = True
    allow_fractional_shares: bool = False
    cash_limited_buy: bool = False
    rsi_high_threshold: float = Field(default=65, ge=0, le=100)
    rsi_mid_high: float = Field(default=60, ge=0, le=100)
    rsi_neutral: float = Field(default=50, ge=0, le=100)
    rsi_mid_low: float = Field(default=40, ge=0, le=100)
    rsi_low_threshold: float = Field(default=35, ge=0, le=100)
    ma_short_period: int = Field(default=3, ge=1, le=100)
    ma_long_period: int = Field(default=7, ge=2, le=200)
    roc_period: int = Field(default=4, ge=1, le=52)
    btc_ticker: str = "BTC-USD"
    btc_lookback_days: int = Field(default=1, ge=1, le=30)
    btc_threshold_pct: float = Field(default=0, ge=0, le=100)
    defense: ModeSettings = ModeSettings(
        slices=7,
        buy_cond_pct=3,
        tp_pct=0.2,
        max_hold_days=30,
    )
    offense: ModeSettings = ModeSettings(
        slices=7,
        buy_cond_pct=5,
        tp_pct=2.5,
        max_hold_days=7,
    )

    @field_validator("target_ticker", "momentum_ticker", "btc_ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not TICKER_PATTERN.fullmatch(ticker):
            raise ValueError("지원하지 않는 티커 형식입니다.")
        return ticker

    @model_validator(mode="after")
    def validate_strategy(self) -> StrategySettings:
        if self.mode_switch_strategy == "ma_cross" and self.ma_short_period >= self.ma_long_period:
            raise ValueError("ma_short_period는 ma_long_period보다 작아야 합니다.")
        if not (
            self.rsi_low_threshold
            <= self.rsi_mid_low
            <= self.rsi_neutral
            <= self.rsi_mid_high
            <= self.rsi_high_threshold
        ):
            raise ValueError("RSI 임계값은 low부터 high까지 오름차순이어야 합니다.")
        return self


class BacktestRequest(BaseModel):
    start_date: date
    end_date: date
    initial_cash: float = Field(default=10_000, gt=0, le=100_000_000)
    log_scale: bool = True
    spread_buy_levels: int = Field(default=5, ge=0, le=20)
    spread_buy_step: int = Field(default=1, ge=1, le=1000)
    strategy: StrategySettings

    @model_validator(mode="after")
    def validate_period(self) -> BacktestRequest:
        if self.start_date >= self.end_date:
            raise ValueError("start_date는 end_date보다 이전이어야 합니다.")
        if (self.end_date - self.start_date).days > 15 * 366:
            raise ValueError("공개 viewer의 최대 조회 기간은 15년입니다.")
        return self
