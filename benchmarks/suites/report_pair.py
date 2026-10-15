"""Pareto fronts of `search_pair` studies, evaluated as a mix with the live strategy, plus ±10% neighbours.

usage: report_pair FAMILY [FAMILY ...] [--prefix mix_]   (studies PREFIXFAMILY) → results/PREFIXresults.json
Saved output: results/mix_report.txt, results/mix2_report.txt (--prefix mix2_)
"""
import argparse
import json
import random
from multiprocessing import Pool

import numpy as np

from benchmarks.lib import candidates
from benchmarks.lib.search import RESULTS_DIR, pareto_front, perturb, strategy
from benchmarks.lib.windows import windows

WIN = windows('test23~', '21~real', '2022', '11~20', '11~all')
_cur = {}


def cur_curve(w):
    if w not in _cur:
        fn, D, O = candidates.current()
        eq = WIN[w].run('SOXL', fn, D, O)
        _cur[w] = eq / eq.iloc[0]
    return _cur[w]


def evaluate(args):
    fam, prm, names = args
    wc = prm['w_cur']; out = {}
    fn, D, O = strategy(fam, prm)
    for w in names:
        eq = WIN[w].run('SOXL', fn, D, O)
        eq = eq / eq.iloc[0]
        out[w] = WIN[w].stat(wc * cur_curve(w) + (1 - wc) * eq)
        out[w + ' solo'] = WIN[w].stat(eq)
    return out


def main(argv):
    ap = argparse.ArgumentParser(prog='report_pair')
    ap.add_argument('families', nargs='+')
    ap.add_argument('--prefix', default='mix_')
    a = ap.parse_args(argv)
    with Pool() as p:
        print('CURRENT solo', {w: tuple(round(x * 100, 1) for x in WIN[w].stat(cur_curve(w))) for w in WIN})
        rows = []
        for fam in a.families:
            pts, n = pareto_front(f'{a.prefix}{fam}')
            res = p.map(evaluate, [(fam, t.params, list(WIN)) for t in pts])
            rng = random.Random(2)
            nb_in = []
            for t in pts:
                for _ in range(12):
                    q = perturb({k: v for k, v in t.params.items() if k != 'w_cur'}, rng); q['w_cur'] = t.params['w_cur']
                    nb_in.append((fam, q, ['11~all', '21~real']))
            nb = p.map(evaluate, nb_in)
            print(f'\n== {fam}: {n} trials, front {len(pts)}')
            for i, (t, r) in enumerate(zip(pts, res)):
                z = nb[i * 12:(i + 1) * 12]
                x = np.array([o['11~all'] for o in z]); y = np.array([o['21~real'] for o in z])
                rob = (np.median(x[:, 0]), np.median(x[:, 1]), x[:, 1].min(), np.median(y[:, 0]), np.median(y[:, 1]), y[:, 1].min())
                print(f"{fam}#{i:<3d} w{t.params['w_cur']:.1f} tr {t.values[0]*100:5.1f}/{t.values[1]*100:6.1f} | "
                      + ' | '.join(f'{w} {r[w][0]*100:5.1f}/{r[w][1]*100:6.1f}' for w in WIN)
                      + f" || solo21~ {r['21~real solo'][0]*100:5.1f}/{r['21~real solo'][1]*100:6.1f}"
                      + f' || nb11~ {rob[0]*100:5.1f}/{rob[1]*100:6.1f} w{rob[2]*100:6.1f} nb21~ {rob[3]*100:5.1f}/{rob[4]*100:6.1f} w{rob[5]*100:6.1f}')
                rows.append({'fam': fam, 'i': i, 'params': t.params, 'train': t.values, 'res': r, 'rob': rob})
    (RESULTS_DIR / f'{a.prefix}results.json').write_text(json.dumps(rows, default=float, indent=1))
