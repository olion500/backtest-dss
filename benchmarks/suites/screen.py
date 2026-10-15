"""First-pass signal screen: many simple mode signals with the live mode parameters, SOXL 2011~.

Columns: full period, 2021~, 2022 return and drawdown, pre-2020, mode switches, % of days in offense.
"""
from multiprocessing import Pool

import pandas as pd

from benchmarks.lib import signals as S
from benchmarks.lib.data import load
from benchmarks.lib.engine import current_params, run
from benchmarks.lib.windows import END, metrics

c, D, O = current_params()


def evaluate(target, modes, D, O, start='2011-01-01'):
    eq = run(target, modes, D, O, start, END).equity
    out = {}
    out['cagr'], out['mdd'] = metrics(eq)
    e21 = eq.loc['2021-01-01':]; e21 = e21 / e21.iloc[0]
    out['cagr21'], out['mdd21'] = metrics(e21)
    e22 = eq.loc['2022-01-01':'2022-12-31']
    out['ret22'] = e22.iloc[-1] / eq.loc[:'2021-12-31'].iloc[-1] - 1
    out['mdd22'] = metrics(e22)[1]
    pre = eq.loc[:'2019-12-31']
    out['cagr_pre20'], out['mdd_pre20'] = metrics(pre)
    if isinstance(modes, pd.Series):
        mm = modes.loc[start:END]; out['switches'] = int((mm != mm.shift()).sum() - 1)
        out['off_pct'] = float((mm == 'offense').mean())
    return out


def fmt(name, o):
    return (f"{name:38s} full {o['cagr']*100:6.1f}% / {o['mdd']*100:6.1f}% | 21~ {o['cagr21']*100:6.1f}% / {o['mdd21']*100:6.1f}% "
            f"| 22 {o['ret22']*100:6.1f}% dd{o['mdd22']*100:6.1f}% | pre20 {o['cagr_pre20']*100:5.1f}%/{o['mdd_pre20']*100:6.1f}% "
            f"| sw {o.get('switches','-')} off {o.get('off_pct',0)*100:.0f}%")


def cands():
    yield 'rsi_current_lag1', ('rsi_mode', dict(high=c['rsi_high_threshold'], mid_high=c['rsi_mid_high'], neutral=c['rsi_neutral'], mid_low=c['rsi_mid_low'], low=c['rsi_low_threshold']))
    yield 'rsi_default_lag1', ('rsi_mode', {})
    yield 'always_offense', ('const', 'offense')
    yield 'always_defense', ('const', 'defense')
    for t in ('QQQ', 'SOXX', 'SOXL'):
        for n in (50, 100, 150, 200):
            for b in (0.0, 0.03):
                yield f'sma {t} {n} band{b}', ('sma_trend', dict(ticker=t, n=n, band=b))
    for t in ('QQQ', 'SOXX'):
        for f, s in ((20, 100), (50, 200), (10, 50)):
            yield f'dual_sma {t} {f}/{s}', ('dual_sma', dict(ticker=t, fast=f, slow=s))
    for hi, lo in ((20, 17), (25, 20), (30, 22), (35, 25)):
        yield f'vix {hi}/{lo}', ('vix_regime', dict(hi=hi, lo=lo))
    for r in (0.9, 0.95, 1.0):
        yield f'vix_term <{r}', ('vix_term', dict(ratio=r))
    for hi, lo in ((0.8, 0.6), (1.0, 0.8), (1.2, 0.9)):
        yield f'rv SOXL {hi}/{lo}', ('realized_vol', dict(hi=hi, lo=lo))
    for dd, rec in ((0.3, 0.15), (0.4, 0.2), (0.5, 0.3)):
        yield f'ddbreak SOXL {dd}/{rec}', ('drawdown_breaker', dict(dd=dd, recover=rec))
    for dd, rec in ((0.1, 0.05), (0.15, 0.08), (0.2, 0.1)):
        yield f'ddbreak QQQ {dd}/{rec}', ('drawdown_breaker', dict(ticker='QQQ', dd=dd, recover=rec))


def job(item):
    name, (fn, kw) = item
    idx = load('SOXL').index
    ms = pd.Series(kw, index=idx) if fn == 'const' else getattr(S, fn)(idx, **kw)
    return name, evaluate('SOXL', ms, D, O)


def main(argv):
    with Pool() as p: res = p.map(job, list(cands()))
    res.sort(key=lambda x: -x[1]['cagr'])
    for n, o in res: print(fmt(n, o))
