"""Capital split across independently run strategies (separate accounts)."""
import itertools
from multiprocessing import Pool
from harness import *
from candidates import CANDIDATES
END = '2026-10-09'
PERIODS = {  # name: (start, end, target, scaled, spread step, cash)
    '21~ real': ('2021-01-01', END, 'SOXL', False, 3, 50000.0),
    '2022': ('2022-01-01', '2022-12-31', 'SOXL', False, 3, 50000.0),
    '23~24': ('2023-01-01', '2024-12-31', 'SOXL', False, 3, 50000.0),
    '25~': ('2025-01-01', END, 'SOXL', False, 3, 50000.0),
    '11~20': ('2011-01-01', '2020-12-31', 'SOXL', True, None, 1e9),
    '02~10 synth': ('2002-01-01', '2010-12-31', 'SOXL_X', True, None, 1e9),
    '02~ all': ('2002-01-01', END, 'SOXL_X', True, None, 1e9),
}
def curve(args):
    name, per = args
    fn, D, O = CANDIDATES[name]()
    s, e, tgt, sc, step, cash = PERIODS[per]
    eq = run(tgt, fn(tgt), D, O, s, e, step=step, cash=cash, scale=sc).equity
    return name, per, eq / eq.iloc[0]

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

if __name__ == '__main__':
    names = list(CANDIDATES)
    with Pool(12) as p: res = p.map(curve, [(n, per) for n in names for per in PERIODS])
    C = {(n, per): eq for n, per, eq in res}
    mixes = [((n,), (1.0,)) for n in names]
    mixes += [(('current', x), (0.5, 0.5)) for x in ('A', 'B', 'C')]
    mixes += [(('current', 'C'), (0.3, 0.7)), (('current', 'C'), (0.7, 0.3)), (('A', 'C'), (0.5, 0.5)),
              (('current', 'A', 'C'), (1/3, 1/3, 1/3))]
    for reb in (False, True):
        print('\n=== rebalance yearly' if reb else '=== no rebalance (separate accounts)')
        for combo, w in mixes:
            if reb and len(combo) == 1: continue
            label = '+'.join(f'{n}{int(round(x*100))}' if len(combo) > 1 else n for n, x in zip(combo, w))
            cells = []
            for per in PERIODS:
                eq = combine([C[(n, per)] for n in combo], w, reb)
                cg, md = metrics(eq)
                if per == '2022': cg = eq.iloc[-1] / eq.iloc[0] - 1
                cells.append(f'{per} {cg*100:5.1f}%/{md*100:5.1f}%')
            print(f'{label:24s}', ' | '.join(cells))
