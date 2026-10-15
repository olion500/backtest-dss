"""Evaluation windows and the metrics reported for them."""
from dataclasses import dataclass

import pandas as pd

from engines.dongpa_engine import summarize

from .engine import run

END = '2026-10-09'


def metrics(eq: pd.Series) -> tuple[float, float]:
    """(CAGR, MDD) as fractions; MDD is negative."""
    m = summarize(eq)
    return m['CAGR'], m['Max Drawdown']


def yearly(eq: pd.Series) -> pd.Series:
    y = eq.resample('YE').last()
    prev = pd.concat([pd.Series([eq.iloc[0]], index=[eq.index[0]]), y.iloc[:-1]])
    return pd.Series(y.values / prev.values - 1, index=y.index.year)


@dataclass(frozen=True)
class Window:
    """real: actual prices, $50k, spread step 3 (how the account trades).
    Otherwise prices are scaled (see data.scaled) with $1e9 so cent rounding and integer
    shares don't distort long histories. synth: the synthetic ``<ticker>_X`` series (2002~).
    total_return: report the period's total return instead of CAGR (single-year windows)."""
    start: str
    end: str = END
    real: bool = False
    synth: bool = False
    total_return: bool = False

    def ticker(self, base: str) -> str:
        return base + '_X' if self.synth else base

    @property
    def step(self):
        return 3 if self.real else None

    @property
    def cash(self) -> float:
        return 50000.0 if self.real else 1e9

    def run(self, base, mode_fn, defense, offense) -> pd.Series:
        """Equity curve; ``mode_fn(ticker)`` gives the daily mode series for that ticker."""
        t = self.ticker(base)
        return run(t, mode_fn(t), defense, offense, self.start, self.end,
                   step=self.step, cash=self.cash, scale=not self.real).equity

    def stat(self, eq: pd.Series) -> tuple[float, float]:
        cagr, mdd = metrics(eq)
        return (eq.iloc[-1] / eq.iloc[0] - 1 if self.total_return else cagr), mdd


WINDOWS = {
    'test23~': Window('2023-01-01', real=True),
    '21~real': Window('2021-01-01', real=True),
    '2022': Window('2022-01-01', '2022-12-31', real=True, total_return=True),
    '23~24': Window('2023-01-01', '2024-12-31', real=True),
    '25~': Window('2025-01-01', real=True),
    '11~20': Window('2011-01-01', '2020-12-31'),
    '11~all': Window('2011-01-01'),
    'syn02~10': Window('2002-01-01', '2010-12-31', synth=True),
    'syn02~': Window('2002-01-01', synth=True),
}


def windows(*names: str) -> dict[str, Window]:
    return {n: WINDOWS[n] for n in names}
