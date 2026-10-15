"""Pareto fronts of `search_mo` studies on later / real-condition windows, plus ±10% neighbour robustness.

usage: report_mo FAMILY [FAMILY ...]   (studies mo_FAMILY) → results/mo_results.json
Saved output: results/mo_report.txt
"""
import json
import random
from multiprocessing import Pool

import numpy as np

from benchmarks.lib import candidates
from benchmarks.lib.search import RESULTS_DIR, pareto_front, perturb, strategy
from benchmarks.lib.windows import windows

WIN = windows('test23~', '21~real', '2022', '11~20', '11~all', 'syn02~10')


def evaluate(args):
    fam, prm, names = args
    fn, D, O = strategy(fam, prm)
    return {w: WIN[w].stat(WIN[w].run('SOXL', fn, D, O)) for w in names}


def fmt(o):
    return ' | '.join(f'{w} {o[w][0]*100:6.1f}%/{o[w][1]*100:6.1f}%' for w in o)


def main(argv):
    fn, D, O = candidates.current()
    cur = {w: win.stat(win.run('SOXL', fn, D, O)) for w, win in WIN.items()}
    print('CURRENT'.ljust(30), fmt(cur))
    rows = []
    with Pool() as p:
        for fam in argv:
            pts, n = pareto_front(f'mo_{fam}')
            res = p.map(evaluate, [(fam, t.params, list(WIN)) for t in pts])
            rng = random.Random(1)
            nb = p.map(evaluate, [(fam, perturb(t.params, rng), ['11~all', '21~real']) for t in pts for _ in range(12)])
            print(f'\n== {fam}: {n} trials, front {len(pts)}')
            for i, (t, r) in enumerate(zip(pts, res)):
                nbr = nb[i * 12:(i + 1) * 12]
                nb_all = np.array([x['11~all'] for x in nbr]); nb21 = np.array([x['21~real'] for x in nbr])
                rob = (np.median(nb_all[:, 0]), np.median(nb_all[:, 1]), nb_all[:, 1].min(), np.median(nb21[:, 0]), np.median(nb21[:, 1]))
                print(f'{fam}#{i:<3d} train {t.values[0]*100:5.1f}%/{t.values[1]*100:6.1f}% | {fmt(r)} '
                      f'|| nb11~ {rob[0]*100:5.1f}%/{rob[1]*100:6.1f}% worst {rob[2]*100:6.1f}% nb21~ {rob[3]*100:5.1f}%/{rob[4]*100:6.1f}%')
                rows.append({'fam': fam, 'i': i, 'params': t.params, 'train': t.values, 'res': r, 'rob': rob})
    (RESULTS_DIR / 'mo_results.json').write_text(json.dumps({'current': cur, 'rows': rows}, default=float, indent=1))
