"""Named strategies for the comparison suites: name -> () -> (mode_fn, defense, offense).

``mode_fn(ticker)`` returns the daily mode series for that ticker. Add a candidate by adding an
entry here; every suite that takes candidate names can then use it.
"""
import json

from . import signals as S
from .data import load
from .engine import current_params
from .search import RESULTS_DIR, strategy


def current():
    """The live RSI strategy from config/strategy.json (signal lagged one day, as the engine trades)."""
    c, D, O = current_params()
    fn = lambda t: S.rsi_mode(load(t).index, c['rsi_high_threshold'], c['rsi_mid_high'], c['rsi_neutral'],
                              c['rsi_mid_low'], c['rsi_low_threshold'])
    return fn, D, O


def robust_params(tag, rank):
    rows = json.loads((RESULTS_DIR / f'{tag}.robust.json').read_text())
    return [r for r in rows if r['rank_in_study'] == rank][0]['params']


def from_study(fam, tag, rank):
    return strategy(fam, robust_params(tag, rank))


def trend_vix():
    """The trend_vix preset (config/presets/trend_vix_2026-10-12.json) as its research signal."""
    return strategy('sma_vix', json.loads((RESULTS_DIR / 'seed_sma_vix0.json').read_text()))


CANDIDATES = {
    'current': current,
    'trend_vix': trend_vix,
    'A': lambda: from_study('sma_vix', 'full_SOXL_X_sma_vix', 1),
    'B': lambda: from_study('sma_vix', 'full_SOXL_X_sma_vix', 0),
    'C': lambda: from_study('ddq', 'full_SOXL_X_ddq', 1),
}
