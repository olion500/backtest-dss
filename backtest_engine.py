"""Simple backtest engine for evaluating parameter combinations locally."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class StrategyParameters:
    """Container holding the four strategy dials used by the optimiser UI."""

    divisions: int
    max_hold_days: int
    buy_threshold_pct: float
    sell_threshold_pct: float


@dataclass(frozen=True)
class BacktestSettings:
    """Runtime settings that apply to both safe and aggressive legs."""

    initial_capital: float
    commission_rate: float = 0.0  # Stored as a percentage (e.g. 0.044 => 0.044%).
    trading_days_per_year: int = 252


@dataclass
class BacktestMetrics:
    """Summarised results from one full simulation run."""

    final_capital: float
    cagr_pct: float
    max_drawdown_pct: float
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class _Position:
    entry_price: float
    units: float
    days_held: int = 0


@dataclass(frozen=True)
class EntryGatePair:
    """Precomputed booleans controlling whether new entries are allowed."""

    safe: Sequence[bool]
    aggressive: Sequence[bool]

    def __post_init__(self) -> None:
        if len(self.safe) != len(self.aggressive):
            raise ValueError("Safe and aggressive gate sequences must share the same length")


class _StrategyLedger:
    """Tracks cash and open positions for a single strategy leg."""

    def __init__(self, params: StrategyParameters, starting_cash: float, commission_rate: float) -> None:
        self.params = params
        self.cash = float(starting_cash)
        self.commission_rate = float(commission_rate)
        self._positions: List[_Position] = []

    def handle_day(self, price: float, previous_price: float | None, *, allow_entry: bool = True) -> None:
        """Update holdings based on the latest close price."""
        self._age_and_close_positions(price)
        if previous_price is None:
            return
        if not allow_entry:
            return
        drop_pct = self._percentage_change(previous_price, price) * -1.0
        if drop_pct < self.params.buy_threshold_pct:
            return
        if len(self._positions) >= self.params.divisions:
            return
        slots_remaining = self.params.divisions - len(self._positions)
        allocation = self.cash / slots_remaining if slots_remaining else 0.0
        if allocation <= 0:
            return
        fee = allocation * self.commission_rate
        spend = allocation - fee
        if spend <= 0:
            return
        units = spend / price
        self.cash -= allocation
        self._positions.append(_Position(entry_price=price, units=units))

    def _age_and_close_positions(self, price: float) -> None:
        for position in list(self._positions):
            position.days_held += 1
            gain_pct = self._percentage_change(position.entry_price, price)
            exit_due_to_gain = gain_pct >= self.params.sell_threshold_pct
            exit_due_to_time = position.days_held >= self.params.max_hold_days
            if not (exit_due_to_gain or exit_due_to_time):
                continue
            proceeds = position.units * price
            fee = proceeds * self.commission_rate
            self.cash += proceeds - fee
            self._positions.remove(position)

    @staticmethod
    def _percentage_change(previous: float, current: float) -> float:
        if previous == 0:
            return 0.0
        return ((current - previous) / previous) * 100.0

    def total_value(self, mark_price: float) -> float:
        position_value = sum(pos.units * mark_price for pos in self._positions)
        return self.cash + position_value


class BacktestEngine:
    """Runs paired safe/aggressive strategies on a shared price series."""

    def __init__(self, prices: Sequence[float]) -> None:
        series = list(prices)
        if len(series) < 2:
            raise ValueError("Need at least two price points to run a backtest")
        if any(price <= 0 for price in series):
            raise ValueError("Prices must all be positive numbers")
        self.prices = series

    def run(
        self,
        settings: BacktestSettings,
        safe_params: StrategyParameters,
        aggressive_params: StrategyParameters,
        *,
        entry_gates: Optional[EntryGatePair] = None,
    ) -> BacktestMetrics:
        safe_cash = settings.initial_capital / 2.0
        aggressive_cash = settings.initial_capital - safe_cash
        commission_fraction = settings.commission_rate / 100.0
        safe_ledger = _StrategyLedger(safe_params, safe_cash, commission_fraction)
        aggressive_ledger = _StrategyLedger(aggressive_params, aggressive_cash, commission_fraction)

        equity_curve: List[float] = []
        safe_gates: Optional[Sequence[bool]] = None
        aggressive_gates: Optional[Sequence[bool]] = None
        if entry_gates is not None:
            if len(entry_gates.safe) != len(self.prices):
                raise ValueError("Safe gate sequence length must match price series length")
            if len(entry_gates.aggressive) != len(self.prices):
                raise ValueError("Aggressive gate sequence length must match price series length")
            safe_gates = entry_gates.safe
            aggressive_gates = entry_gates.aggressive
        previous_price: float | None = None
        for price in self.prices:
            index = len(equity_curve)
            safe_allow = True if safe_gates is None else bool(safe_gates[index])
            aggressive_allow = True if aggressive_gates is None else bool(aggressive_gates[index])
            safe_ledger.handle_day(price, previous_price, allow_entry=safe_allow)
            aggressive_ledger.handle_day(price, previous_price, allow_entry=aggressive_allow)
            total_value = safe_ledger.total_value(price) + aggressive_ledger.total_value(price)
            equity_curve.append(total_value)
            previous_price = price

        final_capital = equity_curve[-1]
        cagr_pct = _compute_cagr(
            initial=settings.initial_capital,
            final=final_capital,
            trading_days=len(self.prices) - 1,
            trading_days_per_year=settings.trading_days_per_year,
        )
        max_drawdown_pct = _compute_max_drawdown(equity_curve)
        return BacktestMetrics(
            final_capital=final_capital,
            cagr_pct=cagr_pct,
            max_drawdown_pct=max_drawdown_pct,
            equity_curve=equity_curve,
        )


def _compute_cagr(
    *,
    initial: float,
    final: float,
    trading_days: int,
    trading_days_per_year: int,
) -> float:
    if initial <= 0 or final <= 0 or trading_days <= 0 or trading_days_per_year <= 0:
        return 0.0
    total_return = final / initial
    years = trading_days / trading_days_per_year
    if years <= 0:
        return 0.0
    return (total_return ** (1.0 / years) - 1.0) * 100.0


def _compute_max_drawdown(equity_curve: Iterable[float]) -> float:
    peak = None
    max_drawdown = 0.0
    for value in equity_curve:
        if value <= 0:
            continue
        if peak is None or value > peak:
            peak = value
            continue
        drawdown = (peak - value) / peak * 100.0
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown
