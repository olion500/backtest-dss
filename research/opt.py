"""Walk-forward Optuna: optimise signal + both modes on TRAIN, report TEST.
usage: opt.py <family> <target> <n_trials> <train_start> <train_end> <tag>"""
import sys, os, json, pickle, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from harness import *
import signals as S
optuna.logging.set_verbosity(optuna.logging.WARNING)
MAX_DD = -0.45

def suggest_mode(t, p):
    sl_on = t.suggest_categorical(f'{p}_sl_on', [False, True])
    return mp(t.suggest_int(f'{p}_slices', 2, 15),
              t.suggest_float(f'{p}_buy', 0.0, 10.0, step=0.25),
              t.suggest_float(f'{p}_tp', 0.5, 10.0, step=0.25),
              t.suggest_int(f'{p}_hold', 3, 60),
              t.suggest_float(f'{p}_sl', 5.0, 40.0, step=1.0) if sl_on else 0)

def mode_from_params(prm, p):
    return mp(prm[f'{p}_slices'], prm[f'{p}_buy'], prm[f'{p}_tp'], prm[f'{p}_hold'],
              prm.get(f'{p}_sl', 0) if prm.get(f'{p}_sl_on') else 0)

def suggest_signal(t, fam, idx):
    if fam == 'single':
        return pd.Series('offense', index=idx)
    if fam == 'rsi':
        lo = t.suggest_float('low', 15, 45, step=0.5); ml = t.suggest_float('mid_low', 30, 55, step=0.5)
        ne = t.suggest_float('neutral', 40, 60, step=0.5); mh = t.suggest_float('mid_high', 50, 75, step=0.5)
        hi = t.suggest_float('high', 55, 85, step=0.5)
        return S.rsi_mode(idx, hi, mh, ne, ml, lo, momo=t.suggest_categorical('momo', ['QQQ', 'SOXX']))
    if fam == 'sma':
        return S.sma_trend(idx, t.suggest_categorical('ticker', ['QQQ', 'SOXX', 'SOXL']),
                           t.suggest_int('n', 20, 250, step=10), t.suggest_float('band', 0.0, 0.06, step=0.01))
    if fam == 'vix':
        lo = t.suggest_int('lo', 12, 35); hi = t.suggest_int('hi_gap', 0, 15) + lo
        return S.vix_regime(idx, hi, lo)
    if fam == 'sma_vix':
        a = S.sma_trend(idx, t.suggest_categorical('ticker', ['QQQ', 'SOXX', 'SOXL']),
                        t.suggest_int('n', 20, 250, step=10), t.suggest_float('band', 0.0, 0.06, step=0.01))
        lo = t.suggest_int('lo', 15, 40); hi = t.suggest_int('hi_gap', 0, 15) + lo
        return S.combine_and(a, S.vix_regime(idx, hi, lo))
    if fam == 'rsi_vix':
        lo = t.suggest_float('low', 15, 45, step=0.5); ml = t.suggest_float('mid_low', 30, 55, step=0.5)
        ne = t.suggest_float('neutral', 40, 60, step=0.5); mh = t.suggest_float('mid_high', 50, 75, step=0.5)
        hi = t.suggest_float('high', 55, 85, step=0.5)
        a = S.rsi_mode(idx, hi, mh, ne, ml, lo, momo=t.suggest_categorical('momo', ['QQQ', 'SOXX']))
        vlo = t.suggest_int('lo', 15, 40); vhi = t.suggest_int('hi_gap', 0, 15) + vlo
        return S.combine_and(a, S.vix_regime(idx, vhi, vlo))
    if fam == 'sma_dd':
        a = S.sma_trend(idx, t.suggest_categorical('ticker', ['SOXX', 'SOXL']),
                        t.suggest_int('n', 20, 250, step=10), t.suggest_float('band', 0.0, 0.06, step=0.01))
        dd = t.suggest_float('dd', 0.05, 0.3, step=0.01); rec = t.suggest_float('rec_frac', 0.2, 1.0, step=0.05) * dd
        return S.combine_and(a, S.drawdown_breaker(idx, t.suggest_categorical('dd_ticker', ['QQQ', 'SOXX']), 252, dd, rec))
    if fam == 'vix_dd':
        lo = t.suggest_int('lo', 15, 40); hi = t.suggest_int('hi_gap', 0, 15) + lo
        dd = t.suggest_float('dd', 0.05, 0.3, step=0.01); rec = t.suggest_float('rec_frac', 0.2, 1.0, step=0.05) * dd
        return S.combine_and(S.vix_regime(idx, hi, lo),
                             S.drawdown_breaker(idx, t.suggest_categorical('ticker', ['QQQ', 'SOXX']), 252, dd, rec))
    if fam == 'rsi_sma':
        lo = t.suggest_float('low', 15, 45, step=0.5); ml = t.suggest_float('mid_low', 30, 55, step=0.5)
        ne = t.suggest_float('neutral', 40, 60, step=0.5); mh = t.suggest_float('mid_high', 50, 75, step=0.5)
        hi = t.suggest_float('high', 55, 85, step=0.5)
        a = S.rsi_mode(idx, hi, mh, ne, ml, lo, momo=t.suggest_categorical('momo', ['QQQ', 'SOXX']))
        b = S.sma_trend(idx, t.suggest_categorical('ticker', ['QQQ', 'SOXX', 'SOXL']),
                        t.suggest_int('n', 20, 250, step=10), t.suggest_float('band', 0.0, 0.06, step=0.01))
        return S.combine_and(a, b)
    if fam == 'ddq':
        dd = t.suggest_float('dd', 0.05, 0.3, step=0.01); rec = t.suggest_float('rec_frac', 0.2, 1.0, step=0.05) * dd
        return S.drawdown_breaker(idx, t.suggest_categorical('ticker', ['QQQ', 'SOXX']), 252, dd, rec)
    raise ValueError(fam)

def build(t, fam, idx):
    ms = suggest_signal(t, fam, idx)
    O = suggest_mode(t, 'off')
    D = O if fam == 'single' else suggest_mode(t, 'def')
    return ms, D, O

class Fixed:
    """Replays stored params through the same suggest_* code paths."""
    def __init__(self, prm): self.p = prm
    def suggest_int(self, k, *a, **kw): return self.p[k]
    def suggest_float(self, k, *a, **kw): return self.p.get(k, 0.0)
    def suggest_categorical(self, k, ch): return self.p[k]

def score(eq):
    cagr, mdd = metrics(eq)
    return cagr if mdd >= MAX_DD else cagr - 5 * (MAX_DD - mdd)

if __name__ == '__main__':
    fam, target, n, ts, te, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    idx = load(target).index
    def objective(t):
        ms, D, O = build(t, fam, idx)
        r = run(target, ms, D, O, ts, te, step=None)
        cagr, mdd = metrics(r.equity)
        t.set_user_attr('cagr', cagr); t.set_user_attr('mdd', mdd)
        return score(r.equity)
    st = JournalStorage(JournalFileBackend(f'studies/{tag}.log'))
    study = optuna.create_study(study_name=tag, storage=st, direction='maximize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=os.getpid(), n_startup_trials=60, multivariate=True))
    study.optimize(objective, n_trials=n)
