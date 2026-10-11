from multiprocessing import Pool
from evalx import *
import signals as S
c, D, O = cur_params()

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
    if fn == 'const': ms = pd.Series(kw, index=idx)
    else: ms = getattr(S, fn)(idx, **kw)
    return name, evaluate('SOXL', ms, D, O)

if __name__ == '__main__':
    with Pool(12) as p: res = p.map(job, list(cands()))
    res.sort(key=lambda x: -x[1]['cagr'])
    for n, o in res: print(fmt(n, o))
    import pickle; pickle.dump(res, open('screen1.pkl', 'wb'))
