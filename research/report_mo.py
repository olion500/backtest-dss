"""report_mo.py fam1 fam2 ...: Pareto fronts (train 2011-2022) evaluated on later / real-condition windows,
plus ±10% neighbourhood robustness for each front point."""
import sys, json, random, numpy as np, optuna
from multiprocessing import Pool
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from harness import *
from opt import build, Fixed
from robust import perturb
END = '2026-10-09'
WIN = {  # name: (start, end, target, scaled, step, cash)
    'test23~': ('2023-01-01', END, 'SOXL', False, 3, 50000.0),
    '21~real': ('2021-01-01', END, 'SOXL', False, 3, 50000.0),
    '2022': ('2022-01-01', '2022-12-31', 'SOXL', False, 3, 50000.0),
    '11~20': ('2011-01-01', '2020-12-31', 'SOXL', True, None, 1e9),
    '11~all': ('2011-01-01', END, 'SOXL', True, None, 1e9),
    'syn02~10': ('2002-01-01', '2010-12-31', 'SOXL_X', True, None, 1e9),
}

def evaluate(args):
    fam, prm, wins = args
    out = {}
    for w in wins:
        s, e, tgt, sc, step, cash = WIN[w]
        ms, D, O = build(Fixed(prm), fam, load(tgt).index)
        eq = run(tgt, ms, D, O, s, e, step=step, cash=cash, scale=sc).equity
        c, m = metrics(eq)
        out[w] = (eq.iloc[-1] / eq.iloc[0] - 1 if w == '2022' else c, m)
    return out

def front(tag):
    st = optuna.load_study(study_name=tag, storage=JournalStorage(JournalFileBackend(f'studies/{tag}.log')))
    seen, pts = set(), []
    for t in st.best_trials:
        k = json.dumps(t.params, sort_keys=True)
        if k in seen or t.values[1] < -0.5: continue
        seen.add(k); pts.append(t)
    return sorted(pts, key=lambda t: t.values[1]), len(st.trials)

def fmt(o):
    return ' | '.join(f'{w} {o[w][0]*100:6.1f}%/{o[w][1]*100:6.1f}%' for w in o)

if __name__ == '__main__':
    import candidates
    fn, D, O = candidates.current()
    from opt import Fixed as _F
    with Pool(12) as p:
        # current strategy, same windows
        cur = {}
        for w, (s, e, tgt, sc, step, cash) in WIN.items():
            eq = run(tgt, fn(tgt), D, O, s, e, step=step, cash=cash, scale=sc).equity
            c, m = metrics(eq); cur[w] = (eq.iloc[-1] / eq.iloc[0] - 1 if w == '2022' else c, m)
        print('CURRENT'.ljust(30), fmt(cur))
        allrows = []
        for fam in sys.argv[1:]:
            pts, n = front(f'mo_{fam}')
            res = p.map(evaluate, [(fam, t.params, list(WIN)) for t in pts])
            rng = random.Random(1)
            nb_jobs = [(fam, perturb(t.params, rng), ['11~all', '21~real']) for t in pts for _ in range(12)]
            nb = p.map(evaluate, nb_jobs)
            print(f'\n== {fam}: {n} trials, front {len(pts)}')
            for i, (t, r) in enumerate(zip(pts, res)):
                nbr = nb[i * 12:(i + 1) * 12]
                nb_all = np.array([x['11~all'] for x in nbr]); nb21 = np.array([x['21~real'] for x in nbr])
                rob = (np.median(nb_all[:, 0]), np.median(nb_all[:, 1]), nb_all[:, 1].min(), np.median(nb21[:, 0]), np.median(nb21[:, 1]))
                print(f'{fam}#{i:<3d} train {t.values[0]*100:5.1f}%/{t.values[1]*100:6.1f}% | {fmt(r)} '
                      f'|| nb11~ {rob[0]*100:5.1f}%/{rob[1]*100:6.1f}% worst {rob[2]*100:6.1f}% nb21~ {rob[3]*100:5.1f}%/{rob[4]*100:6.1f}%')
                allrows.append({'fam': fam, 'i': i, 'params': t.params, 'train': t.values, 'res': r, 'rob': rob})
        json.dump({'current': cur, 'rows': allrows}, open('studies/mo_results.json', 'w'), default=float, indent=1)
