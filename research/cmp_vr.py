"""Compare VR (vr.py) with the current strategy and buy & hold on the same windows."""
from multiprocessing import Pool
from harness import *
from vr import run_vr
import candidates
END = '2026-10-09'
WIN = {  # name: (start, end, real data?)  real → $50k unscaled, else scaled $1e9
    '11~all': ('2011-01-01', END, False),
    '11~20': ('2011-01-01', '2020-12-31', False),
    '21~real': ('2021-01-01', END, True),
    '2022': ('2022-01-01', '2022-12-31', True),
    '23~real': ('2023-01-01', END, True),
    '02~synth': ('2002-01-01', END, False),
}

def stat(eq, w):
    eq = eq / eq.iloc[0]; cg, md = metrics(eq)
    return (eq.iloc[-1] - 1 if w == '2022' else cg), md

def frame(tgt, w):
    s, e, real = WIN[w]
    if w == '02~synth': tgt = tgt + '_X'
    return (load(tgt) if real else scaled(tgt)).loc[s:e], real

def job(a):
    kind, tgt, cfg = a; out = {}
    for w in WIN:
        df, real = frame(tgt, w); cash = 50000.0 if real else 1e9
        if kind == 'current':
            fn, D, O = candidates.current(); t = tgt + ('_X' if w == '02~synth' else '')
            eq = run(t, fn(t), D, O, WIN[w][0], WIN[w][1], step=3 if real else None, cash=cash, scale=not real).equity
        elif kind == 'hold':
            eq = df['Close']
        else:
            eq = run_vr(df, cash, **cfg)
        out[w] = stat(eq, w)
    return kind, tgt, cfg, out

if __name__ == '__main__':
    jobs = []
    for tgt in ('SOXL', 'TQQQ'):
        jobs += [('current', tgt, {}), ('hold', tgt, {})]
        for adv in (False, True):
            for G in (10, 20, 40):
                for fill in ('close', 'limit'):
                    jobs.append(('vr', tgt, dict(G=G, advanced=adv, fill=fill)))
        for sf in (0.3, 0.7):
            jobs.append(('vr', tgt, dict(G=10, advanced=True, stock_frac=sf)))
        for b in (0.10, 0.05):
            jobs.append(('vr', tgt, dict(G=10, advanced=True, band=b)))
        jobs.append(('vr', tgt, dict(G=10, advanced=True, pool_limit=0.75)))
    with Pool(12) as p: res = p.map(job, jobs)
    print(f"{'':44s}" + ''.join(f'{w:>17s}' for w in WIN))
    for kind, tgt, cfg, out in res:
        lab = f'{tgt} {kind} ' + ' '.join(f'{k}={v}' for k, v in cfg.items())
        print(f'{lab:44s}' + ''.join(f'{out[w][0]*100:7.1f}/{out[w][1]*100:6.1f}%  ' for w in WIN))
