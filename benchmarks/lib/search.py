"""Optuna search spaces (signal families + mode parameters) and study helpers.

A family is a named way to build the daily mode series. ``build()`` draws one strategy from a
trial; ``Fixed`` replays stored params through the same code so results can be re-evaluated.
To add a family, add a branch to ``suggest_signal`` and list it in ``FAMILIES``.
"""
import json
import multiprocessing
import os
import random
from pathlib import Path

import pandas as pd

from . import signals as S
from .engine import mp
from .windows import metrics

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
MAX_DD = -0.45
FAMILIES = ['single', 'rsi', 'sma', 'vix', 'sma_vix', 'rsi_vix', 'sma_dd', 'vix_dd', 'rsi_sma', 'ddq']


class Fixed:
    """Replays stored params through the same suggest_* code paths."""
    def __init__(self, prm): self.p = prm
    def suggest_int(self, k, *a, **kw): return self.p[k]
    def suggest_float(self, k, *a, **kw): return self.p.get(k, 0.0)
    def suggest_categorical(self, k, ch): return self.p[k]


def suggest_mode(t, p, flat_sl=False, max_slices=15):
    """flat_sl: one fixed-shape search space where sl 0 = off (multivariate TPE can model it).
    Stored flat params have no ``<p>_sl_on`` key, so replays detect the shape themselves."""
    if flat_sl or (isinstance(t, Fixed) and f'{p}_sl_on' not in t.p):
        sl = t.suggest_float(f'{p}_sl', 0.0, 40.0, step=1.0)
        return mp(t.suggest_int(f'{p}_slices', 2, max_slices),
                  t.suggest_float(f'{p}_buy', 0.0, 10.0, step=0.25),
                  t.suggest_float(f'{p}_tp', 0.5, 10.0, step=0.25),
                  t.suggest_int(f'{p}_hold', 3, 60), sl if sl >= 3 else 0)
    sl_on = t.suggest_categorical(f'{p}_sl_on', [False, True])
    return mp(t.suggest_int(f'{p}_slices', 2, 15),
              t.suggest_float(f'{p}_buy', 0.0, 10.0, step=0.25),
              t.suggest_float(f'{p}_tp', 0.5, 10.0, step=0.25),
              t.suggest_int(f'{p}_hold', 3, 60),
              t.suggest_float(f'{p}_sl', 5.0, 40.0, step=1.0) if sl_on else 0)


def _rsi(t, idx):
    lo = t.suggest_float('low', 15, 45, step=0.5); ml = t.suggest_float('mid_low', 30, 55, step=0.5)
    ne = t.suggest_float('neutral', 40, 60, step=0.5); mh = t.suggest_float('mid_high', 50, 75, step=0.5)
    hi = t.suggest_float('high', 55, 85, step=0.5)
    return S.rsi_mode(idx, hi, mh, ne, ml, lo, momo=t.suggest_categorical('momo', ['QQQ', 'SOXX']))


def _sma(t, idx, tickers=('QQQ', 'SOXX', 'SOXL')):
    return S.sma_trend(idx, t.suggest_categorical('ticker', list(tickers)),
                       t.suggest_int('n', 20, 250, step=10), t.suggest_float('band', 0.0, 0.06, step=0.01))


def _vix(t, idx, lo_min, lo_max):
    lo = t.suggest_int('lo', lo_min, lo_max); hi = t.suggest_int('hi_gap', 0, 15) + lo
    return S.vix_regime(idx, hi, lo)


def _dd(t, idx, ticker_key):
    dd = t.suggest_float('dd', 0.05, 0.3, step=0.01); rec = t.suggest_float('rec_frac', 0.2, 1.0, step=0.05) * dd
    return S.drawdown_breaker(idx, t.suggest_categorical(ticker_key, ['QQQ', 'SOXX']), 252, dd, rec)


def suggest_signal(t, fam, idx):
    if fam == 'single':
        return pd.Series('offense', index=idx)
    if fam == 'rsi':
        return _rsi(t, idx)
    if fam == 'sma':
        return _sma(t, idx)
    if fam == 'vix':
        return _vix(t, idx, 12, 35)
    if fam == 'sma_vix':
        return S.combine_and(_sma(t, idx), _vix(t, idx, 15, 40))
    if fam == 'rsi_vix':
        return S.combine_and(_rsi(t, idx), _vix(t, idx, 15, 40))
    if fam == 'sma_dd':
        return S.combine_and(_sma(t, idx, ('SOXX', 'SOXL')), _dd(t, idx, 'dd_ticker'))
    if fam == 'vix_dd':
        return S.combine_and(_vix(t, idx, 15, 40), _dd(t, idx, 'ticker'))
    if fam == 'rsi_sma':
        return S.combine_and(_rsi(t, idx), _sma(t, idx))
    if fam == 'ddq':
        return _dd(t, idx, 'ticker')
    raise ValueError(fam)


def build(t, fam, idx, flat_sl=False, max_slices=15):
    """(daily mode series, defense params, offense params)."""
    ms = suggest_signal(t, fam, idx)
    O = suggest_mode(t, 'off', flat_sl, max_slices)
    D = O if fam == 'single' else suggest_mode(t, 'def', flat_sl, max_slices)
    return ms, D, O


def strategy(fam, prm):
    """Stored params → (mode_fn, defense, offense); ``mode_fn(ticker)`` is the daily mode series (for Window.run)."""
    from .data import load
    t = Fixed(prm)
    O = suggest_mode(t, 'off')
    D = O if fam == 'single' else suggest_mode(t, 'def')
    return (lambda ticker: suggest_signal(Fixed(prm), fam, load(ticker).index)), D, O


def score(eq):
    """Single-objective score: CAGR, penalised below the MAX_DD drawdown limit."""
    cagr, mdd = metrics(eq)
    return cagr if mdd >= MAX_DD else cagr - 5 * (MAX_DD - mdd)


INT_KEYS = {'n', 'lo', 'hi_gap', 'off_slices', 'def_slices', 'off_hold', 'def_hold'}


def perturb(prm, rng: random.Random):
    """Every numeric param × U(0.9, 1.1): the ±10% neighbourhood used for robustness checks."""
    q = dict(prm)
    for k, v in prm.items():
        if isinstance(v, bool) or isinstance(v, str): continue
        x = v * rng.uniform(0.9, 1.1)
        if k in INT_KEYS: x = max(1, int(round(x)))
        if k == 'n': x = max(20, x)
        q[k] = x
    return q


# ---------------------- Studies ----------------------

def _storage(tag):
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    RESULTS_DIR.mkdir(exist_ok=True)
    return JournalStorage(JournalFileBackend(str(RESULTS_DIR / f'{tag}.log')))


def load_study(tag):
    import optuna
    return optuna.load_study(study_name=tag, storage=_storage(tag))


def optimize(tag, objective, n_trials, directions, startup=60, seed_params=None):
    """One worker: create/join the study ``tag`` and run ``n_trials``. Workers share the journal file."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=tag, storage=_storage(tag), directions=directions, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=os.getpid(), n_startup_trials=startup,
                                                                   multivariate=True))
    if seed_params and not study.trials:
        study.enqueue_trial(seed_params)
    study.optimize(objective, n_trials=n_trials)


def run_workers(target, args, workers):
    """Run ``target(*args)`` in ``workers`` processes and wait (each process seeds its own sampler)."""
    procs = [multiprocessing.Process(target=target, args=args) for _ in range(workers)]
    for p in procs: p.start()
    for p in procs: p.join()


def pareto_front(tag, mdd_floor=-0.5):
    """Unique Pareto-optimal trials with MDD above ``mdd_floor``, deepest MDD first; plus the trial count."""
    st = load_study(tag)
    seen, pts = set(), []
    for t in st.best_trials:
        k = json.dumps(t.params, sort_keys=True)
        if k in seen or t.values[1] < mdd_floor: continue
        seen.add(k); pts.append(t)
    return sorted(pts, key=lambda t: t.values[1]), len(st.trials)


def feasible_trials(tag):
    """Single-objective trials within MAX_DD, best score first."""
    st = load_study(tag)
    tr = [t for t in st.trials if t.value is not None and t.user_attrs.get('mdd', -1) >= MAX_DD]
    return sorted(tr, key=lambda t: -t.value), len(st.trials)
