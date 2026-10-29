"""Best strategy per MDD cap (40/30/20/10%) over the last 5 / 10 / 15 years, data/ prices.

usage: search_mdd search FAMILY TRIALS_PER_WORKER [--years 5|10|15] [--workers N]   (Pareto CAGR × MDD)
       search_mdd report FAMILY [FAMILY ...] [--years 5|10|15]   → results/mdd_caps_<Y>y.txt, .json
Windows end at the latest data/ bar. 5y (2021-10-25~) trades like the accounts: real prices, $50k,
1-share spread steps. 10y (2016-10-25~) and 15y (2011-10-25~) use scaled prices and $1e9 like the
other long windows (SOXL was ~$1 in 2011, cents would distort). Slices go up to 40 so low-exposure
strategies exist. Report: per cap, the top trials within the cap are re-run with every param
×U(0.9,1.1) 12 times; the pick has the best neighbour median CAGR whose neighbour median MDD stays
within the cap. Every row also shows the other windows and 2022 alone (5y conditions).
2026-10-29 run, per family (rsi sma_vix ddq rsi_vix single) and years: search_mdd search FAMILY 250 --years Y --workers 12
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np

from benchmarks.lib import signals as S
from benchmarks.lib.data import load, use_repo_data
from benchmarks.lib.engine import current_params, run
from benchmarks.lib.search import (FAMILIES, RESULTS_DIR, Fixed, build, load_study, optimize, perturb,
                                   run_workers)
from benchmarks.lib.windows import metrics

STARTS = {5: '2021-10-25', 10: '2016-10-25', 15: '2011-10-25'}
CAPS = (0.40, 0.30, 0.20, 0.10)
MAX_SLICES = 40
TOP, NB = 15, 12


def tag(fam, years):
    return f'mdd_{fam}' if years == 5 else f'mdd{years}y_{fam}'


def window(modes, D, O, years):
    if years == 5:
        return run('SOXL', modes, D, O, STARTS[5], None, step=1, cash=50000.0, scale=False).equity
    return run('SOXL', modes, D, O, STARTS[years], None).equity


def _stats(modes, D, O):
    """(5y CAGR, MDD, 10y CAGR, MDD, 15y CAGR, MDD, 2022 return, 2022 MDD)."""
    out = []
    for y in STARTS:
        eq = window(modes, D, O, y)
        out += metrics(eq)
        if y == 5: e22 = eq.loc['2022-01-01':'2022-12-31']
    return (*out, e22.iloc[-1] / e22.iloc[0] - 1, metrics(e22)[1])


def evaluate(fam, prm):
    use_repo_data()
    return _stats(*build(Fixed(prm), fam, load('SOXL').index, flat_sl=True, max_slices=MAX_SLICES))


def _eval(args):
    return evaluate(*args)


def objective_for(fam, years):
    use_repo_data()
    idx = load('SOXL').index

    def objective(t):
        ms, D, O = build(t, fam, idx, flat_sl=True, max_slices=MAX_SLICES)
        return metrics(window(ms, D, O, years))
    return objective


def worker(a):
    optimize(tag(a.family, a.years), objective_for(a.family, a.years), a.trials, ['maximize', 'maximize'], startup=80)


def current_row():
    """The live strategy, the engine deciding its RSI mode."""
    use_repo_data()
    c, D, O = current_params()
    rsi = {k: c[k] for k in ('rsi_high_threshold', 'rsi_mid_high', 'rsi_neutral', 'rsi_mid_low', 'rsi_low_threshold')}
    return _stats(rsi, D, O)


def fmt(r):
    return ' | '.join(f'{y}y {r[2*i]*100:6.1f}% / {r[2*i+1]*100:6.1f}%' for i, y in enumerate(STARTS)) + \
        f' | 2022 {r[6]*100:6.1f}% (MDD {r[7]*100:6.1f}%)'


def params_text(fam, p):
    def mode(k):
        sl = p.get(f'{k}_sl', 0)
        return (f"N{p[f'{k}_slices']} buy {p[f'{k}_buy']:.2f}% tp {p[f'{k}_tp']:.2f}% hold {p[f'{k}_hold']}"
                + (f' SL {sl:.0f}%' if sl >= 3 else ''))
    sig = {k: v for k, v in p.items() if not k.startswith(('off_', 'def_'))}
    return f"{fam} {sig} | offense {mode('off')}" + ('' if fam == 'single' else f" | defense {mode('def')}")


def report(fams, years):
    k = 2 * list(STARTS).index(years)
    trials = []
    for fam in fams:
        seen = set()
        for t in load_study(tag(fam, years)).trials:
            if t.values is None: continue
            key = json.dumps(t.params, sort_keys=True)
            if key in seen: continue
            seen.add(key); trials.append((fam, t.params, t.values))
    lines = [f'{years}y search ({STARTS[years]}~), trials {len(trials)} ({", ".join(fams)})',
             'CURRENT  ' + fmt(current_row())]
    out = {}
    rng = random.Random(0)
    with Pool() as pool:
        for cap in CAPS:
            ok = sorted([x for x in trials if x[2][1] >= -cap], key=lambda x: -x[2][0])[:TOP]
            base = pool.map(_eval, [(f, p) for f, p, _ in ok])
            nb = pool.map(_eval, [(f, perturb(p, rng)) for f, p, _ in ok for _ in range(NB)])
            rows = []
            for i, ((f, p, _), b) in enumerate(zip(ok, base)):
                n = np.array(nb[i * NB:(i + 1) * NB])
                rows.append({'fam': f, 'params': p, 'base': b, 'nb_cagr': float(np.median(n[:, k])),
                             'nb_mdd': float(np.median(n[:, k + 1])), 'nb_worst': float(n[:, k + 1].min())})
            rows.sort(key=lambda r: -(r['nb_cagr'] if r['nb_mdd'] >= -cap else r['nb_cagr'] - 10))
            lines.append(f'\n== MDD <= {cap*100:.0f}%  (top {len(rows)} in cap, sorted by ±10% neighbour median {years}y)')
            for r in rows:
                lines.append(f"{r['fam']:8s} {fmt(r['base'])} || nb {r['nb_cagr']*100:6.1f}% / {r['nb_mdd']*100:6.1f}% "
                             f"worst {r['nb_worst']*100:6.1f}%")
            if rows: lines.append('PICK ' + params_text(rows[0]['fam'], rows[0]['params']))
            out[f'{cap:.2f}'] = rows
    txt = '\n'.join(lines)
    print(txt)
    (RESULTS_DIR / f'mdd_caps_{years}y.txt').write_text(txt + '\n')
    (RESULTS_DIR / f'mdd_caps_{years}y.json').write_text(json.dumps(out, indent=1, default=float))


def main(argv):
    ap = argparse.ArgumentParser(prog='search_mdd')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('family', choices=FAMILIES)
    s.add_argument('trials', type=int)
    s.add_argument('--workers', type=int, default=os.cpu_count())
    r = sub.add_parser('report')
    r.add_argument('families', nargs='+', choices=FAMILIES)
    for p in (s, r):
        p.add_argument('--years', type=int, choices=list(STARTS), default=5)
    a = ap.parse_args(argv)
    if a.cmd == 'search':
        run_workers(worker, (a,), a.workers)
    else:
        report(a.families, a.years)
