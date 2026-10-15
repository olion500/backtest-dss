"""Neighbourhood robustness of a `search` study's top trials: each param ×U(0.9,1.1), 16 times, 2002~.

usage: robust TAG FAMILY TARGET [TOPN]   → results/TAG.robust.json (read by lib/candidates.py)
"""
import json
import random
from multiprocessing import Pool

import numpy as np

from benchmarks.lib.data import load
from benchmarks.lib.engine import run
from benchmarks.lib.search import RESULTS_DIR, Fixed, build, feasible_trials, perturb
from benchmarks.lib.windows import END, metrics

START = '2002-01-01'


def one(args):
    fam, target, prm = args
    ms, D, O = build(Fixed(prm), fam, load(target).index)
    eq = run(target, ms, D, O, START, END).equity
    c, m = metrics(eq)
    c21, m21 = metrics(eq.loc['2021-01-01':])
    return c, m, c21, m21


def main(argv):
    tag, fam, target = argv[:3]; N = int(argv[3]) if len(argv) > 3 else 15
    tr, _ = feasible_trials(tag)
    seen, top = set(), []
    for t in tr:
        key = json.dumps(t.params, sort_keys=True)
        if key in seen: continue
        seen.add(key); top.append(t)
        if len(top) >= N: break
    rng = random.Random(0)
    jobs, owner = [], []
    for i, t in enumerate(top):
        jobs.append((fam, target, t.params)); owner.append((i, 'base'))
        for _ in range(16): jobs.append((fam, target, perturb(t.params, rng))); owner.append((i, 'nb'))
    with Pool() as p: res = p.map(one, jobs)
    rows = []
    for i, t in enumerate(top):
        base = [r for (j, kind), r in zip(owner, res) if j == i and kind == 'base'][0]
        nb = np.array([r for (j, kind), r in zip(owner, res) if j == i and kind == 'nb'])
        rows.append((i, t, base, np.median(nb[:, 0]), np.median(nb[:, 1]), nb[:, 1].min(), np.median(nb[:, 2]), np.median(nb[:, 3])))
    rows.sort(key=lambda r: -(r[3] if r[4] >= -0.5 else r[3] - 5))
    print(f'== {tag} robustness (base full / nb median full CAGR, nb median MDD, nb worst MDD, nb median 21~)')
    for i, t, b, mc, mm, wm, mc21, mm21 in rows:
        print(f'#{i:<2d} base {b[0]*100:5.1f}%/{b[1]*100:6.1f}% 21~ {b[2]*100:5.1f}%/{b[3]*100:6.1f}% | nb {mc*100:5.1f}% / {mm*100:6.1f}% worst {wm*100:6.1f}% | nb21~ {mc21*100:5.1f}%/{mm21*100:6.1f}%')
    rows_out = [{'rank_in_study': r[0], 'params': r[1].params, 'base': r[2], 'nb_cagr': r[3], 'nb_mdd': r[4], 'nb_worst': r[5]} for r in rows]
    (RESULTS_DIR / f'{tag}.robust.json').write_text(json.dumps(rows_out, indent=1, default=float))
