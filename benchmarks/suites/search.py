"""Walk-forward Optuna: optimise one family's signal + both modes on a train window (CAGR, MDD ≤ 45%).

usage: search FAMILY TARGET TRIALS_PER_WORKER TRAIN_START TRAIN_END TAG [--workers N]
example: search sma_vix SOXL_X 300 2002-01-01 2020-12-31 full_SOXL_X_sma_vix --workers 12
Then: `report TAG FAMILY TARGET` (out of sample) and `robust TAG FAMILY TARGET` (±10% neighbours).
"""
import argparse
import os

from benchmarks.lib.data import load
from benchmarks.lib.engine import run
from benchmarks.lib.search import FAMILIES, build, optimize, run_workers, score
from benchmarks.lib.windows import metrics


def objective_for(fam, target, ts, te):
    idx = load(target).index

    def objective(t):
        ms, D, O = build(t, fam, idx)
        r = run(target, ms, D, O, ts, te, step=None)
        cagr, mdd = metrics(r.equity)
        t.set_user_attr('cagr', cagr); t.set_user_attr('mdd', mdd)
        return score(r.equity)
    return objective


def worker(a):
    optimize(a.tag, objective_for(a.family, a.target, a.train_start, a.train_end), a.trials, ['maximize'])


def main(argv):
    ap = argparse.ArgumentParser(prog='search')
    ap.add_argument('family', choices=FAMILIES)
    for k in ('target', 'trials', 'train_start', 'train_end', 'tag'):
        ap.add_argument(k, type=int if k == 'trials' else str)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    run_workers(worker, (a,), a.workers)
