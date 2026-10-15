"""Top trials of a `search` study evaluated out of sample (2021~, 2022, 2011~).

usage: report TAG FAMILY TARGET [TOPK]
"""
from multiprocessing import Pool

import numpy as np

from benchmarks.lib.data import load
from benchmarks.lib.engine import run
from benchmarks.lib.search import Fixed, build, feasible_trials
from benchmarks.lib.windows import END, metrics

TEST = ('2021-01-01', END)


def eval_params(args):
    fam, target, prm = args
    ms, D, O = build(Fixed(prm), fam, load(target).index)
    out = {}
    for k, (s, e) in {'test': TEST, 'y22': ('2022-01-01', '2022-12-31'), 'full': ('2011-01-01', END)}.items():
        eq = run(target, ms, D, O, s, e, step=None).equity
        out[k] = (eq.iloc[-1] / eq.iloc[0] - 1, metrics(eq)[1]) if k == 'y22' else metrics(eq)
    return out


def main(argv):
    tag, fam, target = argv[:3]; k = int(argv[3]) if len(argv) > 3 else 10
    tr, n = feasible_trials(tag)
    top = tr[:k]
    with Pool() as p: res = p.map(eval_params, [(fam, target, t.params) for t in top])
    print(f'== {tag}: {n} trials, {len(tr)} feasible')
    for i, (t, r) in enumerate(zip(top, res)):
        print(f"#{i:<2d} train {t.user_attrs['cagr']*100:6.1f}%/{t.user_attrs['mdd']*100:6.1f}% | "
              f"TEST21~ {r['test'][0]*100:6.1f}%/{r['test'][1]*100:6.1f}% | 2022 {r['y22'][0]*100:6.1f}% dd{r['y22'][1]*100:6.1f}% | "
              f"full {r['full'][0]*100:6.1f}%/{r['full'][1]*100:6.1f}%")
    tc = np.array([r['test'][0] for r in res]); td = np.array([r['test'][1] for r in res])
    print(f"   top{k} median TEST {np.median(tc)*100:.1f}% / {np.median(td)*100:.1f}%  worst dd {td.min()*100:.1f}%")
    print('   best params:', {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in top[0].params.items()})
