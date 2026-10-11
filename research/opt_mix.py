"""Optimise a strategy to pair with the CURRENT one in separate accounts.
Objectives: (CAGR, MDD) of w*current + (1-w)*candidate over the train window (no rebalancing).
usage: opt_mix.py <family> <n_trials> <train_start> <train_end> <tag> [seed_params.json]"""
import sys, os, json, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from harness import *
from opt import build
import candidates
optuna.logging.set_verbosity(optuna.logging.WARNING)

def current_curve(start, end):
    fn, D, O = candidates.current()
    eq = run('SOXL', fn('SOXL'), D, O, start, end).equity
    return eq / eq.iloc[0]

if __name__ == '__main__':
    fam, n, ts, te, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    idx = load('SOXL').index
    cur = current_curve(ts, te)
    def objective(t):
        w = t.suggest_float('w_cur', 0.3, 0.7, step=0.1)
        ms, D, O = build(t, fam, idx)
        eq = run('SOXL', ms, D, O, ts, te).equity
        mix = w * cur + (1 - w) * eq / eq.iloc[0]
        t.set_user_attr('solo', metrics(eq))
        return metrics(mix)
    st = JournalStorage(JournalFileBackend(f'studies/{tag}.log'))
    study = optuna.create_study(study_name=tag, storage=st, directions=['maximize', 'maximize'], load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=os.getpid(), n_startup_trials=80, multivariate=True))
    if len(sys.argv) > 6 and not study.trials:
        study.enqueue_trial({**json.load(open(sys.argv[6])), 'w_cur': 0.5})
    study.optimize(objective, n_trials=n)
