"""report_mix.py fam1 fam2 ...: Pareto fronts of the pair-with-current search, evaluated as a mix
on later / real-condition windows, with ±10% neighbourhood robustness (weight fixed)."""
import sys, json, random, numpy as np, optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from harness import *
from opt import build, Fixed
from robust import perturb
import candidates
END = '2026-10-09'
import os
PREFIX = os.environ.get('MIX_PREFIX', 'mix_')
WIN = {  # name: (start, end, scaled, step, cash)
    'test23~': ('2023-01-01', END, False, 3, 50000.0),
    '21~real': ('2021-01-01', END, False, 3, 50000.0),
    '2022': ('2022-01-01', '2022-12-31', False, 3, 50000.0),
    '11~20': ('2011-01-01', '2020-12-31', True, None, 1e9),
    '11~all': ('2011-01-01', END, True, None, 1e9),
}
_cur = {}
def cur_curve(w):
    if w not in _cur:
        fn, D, O = candidates.current(); s, e, sc, step, cash = WIN[w]
        eq = run('SOXL', fn('SOXL'), D, O, s, e, step=step, cash=cash, scale=sc).equity
        _cur[w] = eq / eq.iloc[0]
    return _cur[w]

def stat(eq, w):
    c, m = metrics(eq)
    return (eq.iloc[-1] / eq.iloc[0] - 1 if w == '2022' else c, m)

def evaluate(args):
    fam, prm, wins = args
    wc = prm['w_cur']; out = {}
    for w in wins:
        s, e, sc, step, cash = WIN[w]
        ms, D, O = build(Fixed(prm), fam, load('SOXL').index)
        eq = run('SOXL', ms, D, O, s, e, step=step, cash=cash, scale=sc).equity
        eq = eq / eq.iloc[0]
        out[w] = stat(wc * cur_curve(w) + (1 - wc) * eq, w)
        out[w + ' solo'] = stat(eq, w)
    return out

def front(tag):
    st = optuna.load_study(study_name=tag, storage=JournalStorage(JournalFileBackend(f'studies/{tag}.log')))
    seen, pts = set(), []
    for t in st.best_trials:
        k = json.dumps(t.params, sort_keys=True)
        if k in seen or t.values[1] < -0.5: continue
        seen.add(k); pts.append(t)
    return sorted(pts, key=lambda t: t.values[1]), len(st.trials)

if __name__ == '__main__':
    with Pool(12) as p:
        print('CURRENT solo', {w: tuple(round(x * 100, 1) for x in stat(cur_curve(w), w)) for w in WIN})
        rows = []
        for fam in sys.argv[1:]:
            pts, n = front(f'{PREFIX}{fam}')
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
                a = np.array([x['11~all'] for x in z]); b = np.array([x['21~real'] for x in z])
                rob = (np.median(a[:, 0]), np.median(a[:, 1]), a[:, 1].min(), np.median(b[:, 0]), np.median(b[:, 1]), b[:, 1].min())
                print(f"{fam}#{i:<3d} w{t.params['w_cur']:.1f} tr {t.values[0]*100:5.1f}/{t.values[1]*100:6.1f} | "
                      + ' | '.join(f'{w} {r[w][0]*100:5.1f}/{r[w][1]*100:6.1f}' for w in WIN)
                      + f" || solo21~ {r['21~real solo'][0]*100:5.1f}/{r['21~real solo'][1]*100:6.1f}"
                      + f' || nb11~ {rob[0]*100:5.1f}/{rob[1]*100:6.1f} w{rob[2]*100:6.1f} nb21~ {rob[3]*100:5.1f}/{rob[4]*100:6.1f} w{rob[5]*100:6.1f}')
                rows.append({'fam': fam, 'i': i, 'params': t.params, 'train': t.values, 'res': r, 'rob': rob})
        json.dump(rows, open(f'studies/{PREFIX}results.json', 'w'), default=float, indent=1)
