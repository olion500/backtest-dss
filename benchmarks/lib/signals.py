"""Daily mode series. Every signal is computed from closes up to the PREVIOUS trading day
(shift(1)) unless lag=0, so the LOC order for day d is known before the close of d."""
import numpy as np
import pandas as pd

import engines.dongpa_engine as E

from .data import load


def _daily(sig_bool_or_mode, idx, lag):
    s = sig_bool_or_mode
    if s.dtype == bool or s.dtype == np.bool_ or set(pd.unique(s.dropna())) <= {True, False, 0, 1, 0.0, 1.0}:
        s = s.map(lambda v: 'offense' if v else 'defense') if s.notna().all() else s.map(
            lambda v: None if pd.isna(v) else ('offense' if v else 'defense'))
    s = s.reindex(s.index.union(idx)).ffill()
    return s.shift(lag).reindex(idx)

def rsi_mode(idx, high=65, mid_high=60, neutral=50, mid_low=40, low=35, momo='QQQ', period=14, lag=1):
    """The engine's weekly-RSI state machine, as a daily series."""
    w = E.to_weekly_close(load(momo))
    r = E.wilder_rsi(w, period); d = r.diff(); pv = r.shift(1)
    class P: pass
    P.rsi_high_threshold, P.rsi_mid_high, P.rsi_neutral, P.rsi_mid_low, P.rsi_low_threshold = high, mid_high, neutral, mid_low, low
    mode, out = 'defense', {}
    for dt in r.index:
        if pd.isna(r[dt]): out[dt] = mode; continue
        c = E.eval_rsi_conditions(float(r[dt]), float(pv[dt]) if not pd.isna(pv[dt]) else float(r[dt]),
                                  float(d[dt]) if not pd.isna(d[dt]) else 0.0, P)
        mode = c or mode; out[dt] = mode
    s = pd.Series(out)
    # weekly value labelled Friday; daily ffill. lag=0 reproduces the engine (Friday sees its own close)
    full = s.reindex(s.index.union(load(momo).index)).ffill().reindex(load(momo).index)
    return full.shift(lag).reindex(idx).ffill()

def sma_trend(idx, ticker='QQQ', n=200, band=0.0, lag=1):
    """Offense while close > SMA(n)*(1+band); defense below SMA*(1-band); hysteresis in between."""
    c = load(ticker)['Close']; ma = c.rolling(n).mean()
    up, dn = c > ma * (1 + band), c < ma * (1 - band)
    st = pd.Series(np.where(up, 1.0, np.where(dn, 0.0, np.nan)), index=c.index).ffill()
    return _daily(st.dropna().astype(bool), idx, lag)

def dual_sma(idx, ticker='QQQ', fast=50, slow=200, lag=1):
    c = load(ticker)['Close']
    return _daily((c.rolling(fast).mean() > c.rolling(slow).mean()).iloc[slow:], idx, lag)

def vix_regime(idx, hi=25, lo=20, lag=1):
    v = load('VIX')['Close']
    st = pd.Series(np.where(v > hi, 0.0, np.where(v < lo, 1.0, np.nan)), index=v.index).ffill()
    return _daily(st.dropna().astype(bool), idx, lag)

def vix_term(idx, ratio=1.0, lag=1):
    """VIX/VIX3M < ratio (contango) → offense."""
    v, v3 = load('VIX')['Close'], load('VIX3M')['Close']
    r = (v / v3).dropna()
    return _daily(r < ratio, idx, lag)

def realized_vol(idx, ticker='SOXL', n=20, hi=1.0, lo=0.8, lag=1):
    c = load(ticker)['Close']; rv = np.log(c).diff().rolling(n).std() * np.sqrt(252)
    st = pd.Series(np.where(rv > hi, 0.0, np.where(rv < lo, 1.0, np.nan)), index=c.index).ffill()
    return _daily(st.dropna().astype(bool), idx, lag)

def drawdown_breaker(idx, ticker='SOXL', lookback=252, dd=0.4, recover=0.2, lag=1):
    """Defense when ticker is > dd below its rolling high; offense again once within `recover`."""
    c = load(ticker)['Close']; hi = c.rolling(lookback, min_periods=20).max(); x = c / hi - 1
    st = pd.Series(np.where(x < -dd, 0.0, np.where(x > -recover, 1.0, np.nan)), index=c.index).ffill()
    return _daily(st.dropna().astype(bool), idx, lag)

def combine_and(*series):
    """Offense only when every input says offense."""
    df = pd.concat(series, axis=1)
    off = (df == 'offense').all(axis=1)
    out = off.map({True: 'offense', False: 'defense'})
    out[df.isna().any(axis=1)] = None
    return out
