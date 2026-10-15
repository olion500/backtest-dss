"""Multi-objective Optuna on real SOXL: maximise (CAGR, MDD) over a train window.

usage: search_mo FAMILY TRIALS_PER_WORKER TRAIN_START TRAIN_END TAG [--workers N]
2026-10-12 run, per family: search_mo FAMILY 300 2011-01-01 2022-12-31 mo_FAMILY --workers 12
  (families: rsi sma_vix ddq sma_dd vix_dd rsi_sma single). Then: report_mo FAMILY ...
"""
import argparse
import os

from benchmarks.lib.data import load
from benchmarks.lib.engine import run
from benchmarks.lib.search import FAMILIES, build, optimize, run_workers
from benchmarks.lib.windows import metrics


def objective_for(fam, ts, te):
    idx = load('SOXL').index

    def objective(t):
        ms, D, O = build(t, fam, idx)
        eq = run('SOXL', ms, D, O, ts, te).equity
        cagr, mdd = metrics(eq)
        e22 = eq.loc['2022-01-01':'2022-12-31']
        if len(e22): t.set_user_attr('mdd22', metrics(e22)[1])
        return cagr, mdd
    return objective


def worker(a):
    optimize(a.tag, objective_for(a.family, a.train_start, a.train_end), a.trials, ['maximize', 'maximize'], startup=80)


def main(argv):
    ap = argparse.ArgumentParser(prog='search_mo')
    ap.add_argument('family', choices=FAMILIES)
    ap.add_argument('trials', type=int)
    for k in ('train_start', 'train_end', 'tag'):
        ap.add_argument(k)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    run_workers(worker, (a,), a.workers)
