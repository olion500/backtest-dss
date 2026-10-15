"""Run the real engine (``engines/dongpa_engine.py``) with an externally computed daily mode series.

Benchmarks never change the engine: ``run()`` swaps the engine's mode decision for a lookup
in a precomputed series for the duration of one backtest.
"""
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import engines.dongpa_engine as E
from engines.dongpa_engine import CapitalParams, ModeParams, StrategyParams

from .data import load, scaled

STRATEGY_PATH = Path(__file__).resolve().parents[2] / 'config' / 'strategy.json'


def mp(slices, buy, tp, hold, sl=0):
    return ModeParams(buy_cond_pct=buy, tp_pct=tp, max_hold_days=int(hold), slices=int(slices),
                      stop_loss_pct=sl if sl and sl > 0 else None)


def current_params():
    """The live strategy (config/strategy.json): (raw dict, defense, offense)."""
    c = json.loads(STRATEGY_PATH.read_text())
    D = mp(c['defense_slices'], c['defense_buy'], c['defense_tp'], c['defense_hold'], c['defense_sl'])
    O = mp(c['offense_slices'], c['offense_buy'], c['offense_tp'], c['offense_hold'], c['offense_sl'])
    return c, D, O


@contextmanager
def _injected(modes: pd.Series):
    """The engine decides the NEXT session's mode at the end of day d (decide_mode(d)), so the
    injected series (mode FOR each day) is served shifted one session earlier."""
    following = modes.shift(-1)
    decide, initial = E.decide_mode, E.determine_initial_mode

    def _decide(idx, prev_mode, indicators, params):
        v = following.get(idx)
        return v if isinstance(v, str) else (prev_mode or 'defense')

    def _initial(start, indicators, params):
        v = modes.get(start)
        return v if isinstance(v, str) else 'defense'

    E.decide_mode, E.determine_initial_mode = _decide, _initial
    try:
        yield
    finally:
        E.decide_mode, E.determine_initial_mode = decide, initial


def run(target, modes, defense, offense, start, end, step=None, cash=1e9, momo='QQQ', scale=True):
    """modes: pd.Series of 'offense'/'defense' per day, or a dict of StrategyParams fields
    (e.g. RSI thresholds) to let the engine decide the mode itself."""
    t = (scaled(target) if scale else load(target)).loc[start:end]
    m = load(momo).loc[:end]
    p = StrategyParams(target_ticker=target, momentum_ticker=momo, defense=defense, offense=offense,
                       spread_buy_step=step, **(modes if isinstance(modes, dict) else {}))
    if isinstance(modes, pd.Series):
        with _injected(modes.reindex(t.index)):
            return E.run_backtest(t, m, p, CapitalParams(initial_cash=cash))
    return E.run_backtest(t, m, p, CapitalParams(initial_cash=cash))
