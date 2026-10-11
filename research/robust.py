"""robust.py <tag> <fam> <target> [topN]: neighbourhood robustness of top trials (full 2002-2026)."""
import sys, json, random, numpy as np
from multiprocessing import Pool
from report import load_study
from opt import *
START, END = '2002-01-01', '2026-10-09'
INT_KEYS = {'n', 'lo', 'hi_gap', 'off_slices', 'def_slices', 'off_hold', 'def_hold'}

def perturb(prm, rng):
    q = dict(prm)
    for k, v in prm.items():
        if isinstance(v, bool) or isinstance(v, str): continue
        x = v * rng.uniform(0.9, 1.1)
        if k in INT_KEYS: x = max(1, int(round(x)))
        if k == 'n': x = max(20, x)
        q[k] = x
    return q

def one(args):
    fam, target, prm = args
    idx = load(target).index
    ms, D, O = build(Fixed(prm), fam, idx)
    eq = run(target, ms, D, O, START, END).equity
    c, m = metrics(eq)
    e21 = eq.loc['2021-01-01':]; c21, m21 = metrics(e21)
    return c, m, c21, m21

if __name__ == '__main__':
    tag, fam, target = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 15
    st = load_study(tag)
    tr = sorted([t for t in st.trials if t.value is not None and t.user_attrs.get('mdd', -1) >= MAX_DD], key=lambda t: -t.value)
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
    with Pool(12) as p: res = p.map(one, jobs)
    rows = []
    for i, t in enumerate(top):
        base = [r for (j, kind), r in zip(owner, res) if j == i and kind == 'base'][0]
        nb = np.array([r for (j, kind), r in zip(owner, res) if j == i and kind == 'nb'])
        rows.append((i, t, base, np.median(nb[:, 0]), np.median(nb[:, 1]), nb[:, 1].min(), np.median(nb[:, 2]), np.median(nb[:, 3])))
    rows.sort(key=lambda r: -(r[3] if r[4] >= -0.5 else r[3] - 5))
    print(f'== {tag} robustness (base full / nb median full CAGR, nb median MDD, nb worst MDD, nb median 21~)')
    for i, t, b, mc, mm, wm, mc21, mm21 in rows:
        print(f'#{i:<2d} base {b[0]*100:5.1f}%/{b[1]*100:6.1f}% 21~ {b[2]*100:5.1f}%/{b[3]*100:6.1f}% | nb {mc*100:5.1f}% / {mm*100:6.1f}% worst {wm*100:6.1f}% | nb21~ {mc21*100:5.1f}%/{mm21*100:6.1f}%')
    json.dump([{'rank_in_study': r[0], 'params': r[1].params, 'base': r[2], 'nb_cagr': r[3], 'nb_mdd': r[4], 'nb_worst': r[5]} for r in rows],
              open(f'studies/{tag}.robust.json', 'w'), indent=1, default=float)
