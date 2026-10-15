"""Capital split across candidates run in separate accounts, with and without yearly rebalancing.

usage: mix [CANDIDATE ...]   (default: every candidate in lib/candidates.py, plus preset mixes)
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from benchmarks.lib.candidates import CANDIDATES
from benchmarks.lib.windows import windows

WIN = windows('21~real', '2022', '23~24', '25~', '11~20', 'syn02~10', 'syn02~')


def curve(args):
    name, w = args
    fn, D, O = CANDIDATES[name]()
    eq = WIN[w].run('SOXL', fn, D, O)
    return name, w, eq / eq.iloc[0]


def combine(curves, weights, rebalance):
    if not rebalance:
        return sum(w * c for w, c in zip(weights, curves))
    df = pd.concat(curves, axis=1).ffill(); rets = df.pct_change().fillna(0)
    out, val = [], 1.0; hold = np.array(weights, float)
    year = df.index[0].year
    for d, r in rets.iterrows():
        if d.year != year: hold = val * np.array(weights); year = d.year
        hold = hold * (1 + r.values); val = hold.sum(); out.append(val)
    return pd.Series(out, index=df.index)


def main(argv):
    names = argv or list(CANDIDATES)
    with Pool() as p: res = p.map(curve, [(n, w) for n in names for w in WIN])
    C = {(n, w): eq for n, w, eq in res}
    mixes = [((n,), (1.0,)) for n in names]
    if not argv:
        mixes += [(('current', x), (0.5, 0.5)) for x in ('A', 'B', 'C')]
        mixes += [(('current', 'C'), (0.3, 0.7)), (('current', 'C'), (0.7, 0.3)), (('A', 'C'), (0.5, 0.5)),
                  (('current', 'A', 'C'), (1/3, 1/3, 1/3))]
    else:
        mixes += [(tuple(names), tuple(1 / len(names) for _ in names))] if len(names) > 1 else []
    for reb in (False, True):
        print('\n=== rebalance yearly' if reb else '=== no rebalance (separate accounts)')
        for combo, wts in mixes:
            if reb and len(combo) == 1: continue
            label = '+'.join(f'{n}{int(round(x*100))}' if len(combo) > 1 else n for n, x in zip(combo, wts))
            cells = []
            for w, win in WIN.items():
                cg, md = win.stat(combine([C[(n, w)] for n in combo], wts, reb))
                cells.append(f'{w} {cg*100:5.1f}%/{md*100:5.1f}%')
            print(f'{label:24s}', ' | '.join(cells))
