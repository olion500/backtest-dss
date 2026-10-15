"""Price data for benchmarks: ``benchmarks/data/<ticker>.csv`` (gitignored).

``python -m benchmarks fetch`` downloads the raw tickers and ``python -m benchmarks synth``
builds the synthetic pre-listing histories (``SOXL_X``, ``TQQQ_X``, ``USD_X``).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'
TICKERS = ['SOXL', 'QQQ', 'SOXX', 'TQQQ', '^VIX', 'TECL', 'UPRO', 'SPY', 'SMH', '^VIX3M', 'FNGU', 'USD', '^IRX']

_cache: dict[str, pd.DataFrame] = {}


def load(ticker: str) -> pd.DataFrame:
    """Daily OHLCV for ``ticker`` (``^`` dropped from the name: ``VIX``, ``IRX``)."""
    if ticker not in _cache:
        df = pd.read_csv(DATA_DIR / f'{ticker}.csv', parse_dates=['Date'], index_col='Date')
        _cache[ticker] = df[df['Close'].notna()]
    return _cache[ticker]


def scaled(ticker: str) -> pd.DataFrame:
    """Prices scaled so the lowest close is >= $50: removes cent-rounding distortion
    on split-adjusted sub-dollar history. Only ratios matter to the strategy."""
    key = f'{ticker}__scaled'
    if key not in _cache:
        df = load(ticker).copy()
        factor = max(1.0, 50.0 / df['Close'].min())
        for col in ('Open', 'High', 'Low', 'Close', 'Adj Close'):
            if col in df:
                df[col] = df[col] * factor
        _cache[key] = df
    return _cache[key]


def fetch(tickers=TICKERS, start='1999-01-01', end=None):
    import yfinance as yf

    DATA_DIR.mkdir(exist_ok=True)
    for t in tickers:
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False, repair=True)
        if hasattr(df.columns, 'levels'):
            df.columns = df.columns.get_level_values(0)
        df.index.name = 'Date'
        df.to_csv(DATA_DIR / f"{t.replace('^', '')}.csv")
        print(t, len(df), df.index[0].date() if len(df) else None, df.index[-1].date() if len(df) else None)


def synth(base, real, lev=3, er=0.0095, out=None):
    """Synthetic leveraged ETF before listing: lev × base daily return − (expense + (lev−1) × T-bill) / 252.
    ``er`` includes an extra drag calibrated so the overlap with the real fund matches (~0%/yr gap)."""
    b = load(base)
    rf = load('IRX')['Close'].reindex(b.index).ffill().fillna(0) / 100
    r = (lev * b['Adj Close'].pct_change() - (er + (lev - 1) * rf) / 252).fillna(0)

    rl = load(real)['Adj Close'].pct_change()
    ov = rl.dropna().index.intersection(r.index)[5:]
    a, s = (1 + rl[ov]).prod(), (1 + r[ov]).prod()
    print(f'{base}x{lev} vs {real}: corr {np.corrcoef(rl[ov], r[ov])[0, 1]:.4f}  real total x{a:.1f}  synth x{s:.1f}  '
          f'ann diff {((s / a) ** (252 / len(ov)) - 1) * 100:+.2f}%/yr')
    for y in (2011, 2015, 2018, 2020, 2022, 2024):
        m = ov[ov.year == y]
        print(f'   {y}: real {((1 + rl[m]).prod() - 1) * 100:6.1f}%  synth {((1 + r[m]).prod() - 1) * 100:6.1f}%')

    # Splice: back-cast closes before the real fund's first day so that close[t+1]/close[t] = 1 + r[t+1].
    first = load(real).index[0]
    pre = r[r.index < first]
    c_real0 = load(real)['Close'].iloc[0]
    growth_after = np.r_[(1 + pre.values[::-1]).cumprod()[::-1][1:], 1.0] * (1 + r.get(first, 0))
    pre_close = pd.Series(c_real0 / growth_after, index=pre.index)
    d = pd.DataFrame({'Open': pre_close, 'High': pre_close, 'Low': pre_close, 'Close': pre_close,
                      'Adj Close': pre_close, 'Volume': 0})
    full = pd.concat([d, load(real)[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]])
    if out:
        full.index.name = 'Date'
        full.to_csv(DATA_DIR / f'{out}.csv')
    return full
