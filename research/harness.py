"""Research harness: run the real engine with an externally computed daily mode series."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import engines.dongpa_engine as E
from engines.dongpa_engine import ModeParams, CapitalParams, StrategyParams, summarize

DL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_cache = {}
def load(t):
    if t not in _cache:
        df = pd.read_csv(f'{DL}/{t}.csv', parse_dates=['Date'], index_col='Date')
        _cache[t] = df[df['Close'].notna()]
    return _cache[t]

# The engine decides the NEXT session's mode at the end of day d (decide_mode(d)), so the
# injected series (mode FOR each day) is served shifted one session earlier.
_MODES = {'s': None, 'next': None}
_orig_decide, _orig_init = E.decide_mode, E.determine_initial_mode
def _decide(idx, prev_mode, indicators, params):
    s = _MODES['next']
    if s is None: return _orig_decide(idx, prev_mode, indicators, params)
    v = s.get(idx)
    return v if isinstance(v, str) else (prev_mode or 'defense')
def _init(start, indicators, params):
    s = _MODES['s']
    if s is None: return _orig_init(start, indicators, params)
    v = s.get(start); return v if isinstance(v, str) else 'defense'
E.decide_mode = _decide
E.determine_initial_mode = _init

def mp(slices, buy, tp, hold, sl=0):
    return ModeParams(buy_cond_pct=buy, tp_pct=tp, max_hold_days=int(hold), slices=int(slices),
                      stop_loss_pct=sl if sl and sl > 0 else None)

def scaled(t):
    """Target prices scaled so the lowest close is >= $50: removes cent-rounding distortion
    on split-adjusted sub-dollar history. Only ratios matter to the strategy."""
    k = f'{t}__scaled'
    if k not in _cache:
        df = load(t).copy(); f = max(1.0, 50.0 / df['Close'].min())
        for c in ('Open', 'High', 'Low', 'Close', 'Adj Close'):
            if c in df: df[c] = df[c] * f
        _cache[k] = df
    return _cache[k]

def run(target, modes, defense, offense, start, end, step=None, cash=1e9, momo='QQQ', scale=True):
    """modes: pd.Series of 'offense'/'defense' (or None to use engine's RSI with params)."""
    t = (scaled(target) if scale else load(target)).loc[start:end]
    m = load(momo).loc[:end]
    p = StrategyParams(target_ticker=target, momentum_ticker=momo, defense=defense, offense=offense,
                       spread_buy_step=step, **(modes if isinstance(modes, dict) else {}))
    if isinstance(modes, pd.Series):
        s = modes.reindex(t.index)
        _MODES['s'], _MODES['next'] = s, s.shift(-1)
    try:
        r = E.run_backtest(t, m, p, CapitalParams(initial_cash=cash))
    finally:
        _MODES['s'] = _MODES['next'] = None
    return r

def metrics(eq):
    m = summarize(eq)
    return m['CAGR'], m['Max Drawdown']

def yearly(eq):
    y = eq.resample('YE').last()
    first = eq.iloc[0]
    prev = pd.concat([pd.Series([first], index=[eq.index[0]]), y.iloc[:-1]])
    return pd.Series(y.values / prev.values - 1, index=y.index.year)
