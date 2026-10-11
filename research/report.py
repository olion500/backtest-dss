"""report.py <tag> [topk]: train-top trials evaluated out of sample."""
import sys, numpy as np, optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from opt import *
TEST = ('2021-01-01', '2026-10-09')

def load_study(tag):
    return optuna.load_study(study_name=tag, storage=JournalStorage(JournalFileBackend(f'studies/{tag}.log')))

def eval_params(args):
    fam, target, prm = args
    idx = load(target).index
    ms, D, O = build(Fixed(prm), fam, idx)
    out = {}
    for k, (s, e) in {'test': TEST, 'y22': ('2022-01-01', '2022-12-31'), 'full': ('2011-01-01', TEST[1])}.items():
        eq = run(target, ms, D, O, s, e, step=None).equity
        if k == 'y22': out[k] = (eq.iloc[-1] / eq.iloc[0] - 1, metrics(eq)[1])
        else: out[k] = metrics(eq)
    return out

if __name__ == '__main__':
    tag, fam, target = sys.argv[1], sys.argv[2], sys.argv[3]; k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    st = load_study(tag)
    tr = [t for t in st.trials if t.value is not None and t.user_attrs.get('mdd', -1) >= MAX_DD]
    tr.sort(key=lambda t: -t.value)
    top = tr[:k]
    with Pool(12) as p: res = p.map(eval_params, [(fam, target, t.params) for t in top])
    print(f'== {tag}: {len(st.trials)} trials, {len(tr)} feasible')
    for i, (t, r) in enumerate(zip(top, res)):
        print(f"#{i:<2d} train {t.user_attrs['cagr']*100:6.1f}%/{t.user_attrs['mdd']*100:6.1f}% | "
              f"TEST21~ {r['test'][0]*100:6.1f}%/{r['test'][1]*100:6.1f}% | 2022 {r['y22'][0]*100:6.1f}% dd{r['y22'][1]*100:6.1f}% | "
              f"full {r['full'][0]*100:6.1f}%/{r['full'][1]*100:6.1f}%")
    tc = np.array([r['test'][0] for r in res]); td = np.array([r['test'][1] for r in res])
    print(f"   top{k} median TEST {np.median(tc)*100:.1f}% / {np.median(td)*100:.1f}%  worst dd {td.min()*100:.1f}%")
    print('   best params:', {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in top[0].params.items()})
