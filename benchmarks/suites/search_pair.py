"""Optimise a strategy to pair with the live one in a separate account.

Objectives: (CAGR, MDD) of w*current + (1-w)*candidate over the train window (no rebalancing), w in 0.3~0.7.
usage: search_pair FAMILY TRIALS_PER_WORKER TRAIN_START TRAIN_END TAG [--seed PARAMS.json] [--flat-sl] [--workers N]
2026-10-12 runs (TAG mix_FAMILY, 200 trials × 12 workers, train 2011-01-01..2022-12-31):
  sma_vix --seed results/seed_sma_vix0.json, then rsi_sma, vix_dd, ddq;
  mix2_sma_vix: 250 trials, --flat-sl --seed results/seed_sma_vix0_flat.json. Then: report_pair FAMILY ...
"""
import argparse
import json
import os

from benchmarks.lib import candidates
from benchmarks.lib.data import load
from benchmarks.lib.engine import run
from benchmarks.lib.search import FAMILIES, build, optimize, run_workers
from benchmarks.lib.windows import metrics


def current_curve(start, end):
    fn, D, O = candidates.current()
    eq = run('SOXL', fn('SOXL'), D, O, start, end).equity
    return eq / eq.iloc[0]


def worker(a):
    idx = load('SOXL').index
    cur = current_curve(a.train_start, a.train_end)

    def objective(t):
        w = t.suggest_float('w_cur', 0.3, 0.7, step=0.1)
        ms, D, O = build(t, a.family, idx, flat_sl=a.flat_sl)
        eq = run('SOXL', ms, D, O, a.train_start, a.train_end).equity
        t.set_user_attr('solo', metrics(eq))
        return metrics(w * cur + (1 - w) * eq / eq.iloc[0])

    seed = {**json.load(open(a.seed)), 'w_cur': 0.5} if a.seed else None
    optimize(a.tag, objective, a.trials, ['maximize', 'maximize'], startup=80, seed_params=seed)


def main(argv):
    ap = argparse.ArgumentParser(prog='search_pair')
    ap.add_argument('family', choices=FAMILIES)
    ap.add_argument('trials', type=int)
    for k in ('train_start', 'train_end', 'tag'):
        ap.add_argument(k)
    ap.add_argument('--seed')
    ap.add_argument('--flat-sl', action='store_true')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args(argv)
    run_workers(worker, (a,), a.workers)
