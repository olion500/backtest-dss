"""Lump-sum VR (lib/vr.py) vs the live strategy vs buy & hold, SOXL and TQQQ, several windows.

Saved output: results/vr_compare.txt
"""
from multiprocessing import Pool

from benchmarks.lib import candidates
from benchmarks.lib.data import load, scaled
from benchmarks.lib.vr import run_vr
from benchmarks.lib.windows import windows

WIN = windows('11~all', '11~20', '21~real', '2022', 'test23~', 'syn02~')


def job(a):
    kind, tgt, cfg = a; out = {}
    for name, w in WIN.items():
        df = (load if w.real else scaled)(w.ticker(tgt)).loc[w.start:w.end]
        if kind == 'current':
            fn, D, O = candidates.current()
            eq = w.run(tgt, fn, D, O)
        elif kind == 'hold':
            eq = df['Close']
        else:
            eq = run_vr(df, w.cash, **cfg)
        out[name] = w.stat(eq / eq.iloc[0])
    return kind, tgt, cfg, out


def main(argv):
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
    with Pool() as p: res = p.map(job, jobs)
    print(f"{'':44s}" + ''.join(f'{w:>17s}' for w in WIN))
    for kind, tgt, cfg, out in res:
        lab = f'{tgt} {kind} ' + ' '.join(f'{k}={v}' for k, v in cfg.items())
        print(f'{lab:44s}' + ''.join(f'{out[w][0]*100:7.1f}/{out[w][1]*100:6.1f}%  ' for w in WIN))
