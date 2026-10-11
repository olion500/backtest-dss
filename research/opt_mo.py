"""Multi-objective Optuna on real SOXL: maximise (CAGR, MDD) over the train window.
usage: opt_mo.py <family> <n_trials> <train_start> <train_end> <tag>"""
import sys, os, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from harness import *
from opt import build
optuna.logging.set_verbosity(optuna.logging.WARNING)

if __name__ == '__main__':
    fam, n, ts, te, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    idx = load('SOXL').index
    def objective(t):
        ms, D, O = build(t, fam, idx)
        eq = run('SOXL', ms, D, O, ts, te).equity
        cagr, mdd = metrics(eq)
        e22 = eq.loc['2022-01-01':'2022-12-31']
        if len(e22): t.set_user_attr('mdd22', metrics(e22)[1])
        return cagr, mdd
    st = JournalStorage(JournalFileBackend(f'studies/{tag}.log'))
    study = optuna.create_study(study_name=tag, storage=st, directions=['maximize', 'maximize'], load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=os.getpid(), n_startup_trials=80, multivariate=True))
    study.optimize(objective, n_trials=n)
