"""현행 RSI 전략 + 공세 손절 연속 브레이커를 Optuna로 함께 최적화 (최근 5년 학습 → 10·15년·이전 10년 시험).

usage: search_breaker search TRIALS_PER_WORKER [--workers N]   (CAGR × MDD, train 5y)
       search_breaker report                                    → results/search_breaker.txt, .json
Space: the engine's weekly QQQ RSI thresholds, both modes (slices, buy %, tp %, hold, stop %; stop < 3 = off)
and the breaker of lib/transition.py (sl_k offense stops within sl_win sessions → defense for sl_hold;
sl_hold 0 = no breaker). Seeds: the live strategy with and without the recommended breaker (2 / 10 / 5).
Picks use the train window only: the top train Calmar trials are re-run with every number ×U(0.9, 1.1);
best neighbour-median Calmar, and the best neighbour-median CAGR within train MDD 30% / 25%.
Windows and the jitter as in suites/transition.py.
2026-11-01 run: search_breaker search 300 --workers 12, then report
"""
import argparse
import json
import os
import random
from multiprocessing import Pool

import numpy as np
import pandas as pd

from benchmarks.lib.data import use_repo_data
from benchmarks.lib.engine import current_params, mp
from benchmarks.lib.search import RESULTS_DIR, Fixed, load_study, optimize, run_workers
from benchmarks.lib.transition import Policy, run
from benchmarks.lib.windows import metrics
from benchmarks.suites.transition import HEADER, WINDOWS, calmar

TAG = 'search_breaker'
RSI_KEYS = {'low': 'rsi_low_threshold', 'mid_low': 'rsi_mid_low', 'neutral': 'rsi_neutral',
            'mid_high': 'rsi_mid_high', 'high': 'rsi_high_threshold'}
INT_KEYS = {'off_slices', 'def_slices', 'off_hold', 'def_hold', 'sl_k', 'sl_win', 'sl_hold'}
TOP, NB, CAPS, NB_LIVE = 30, 8, (0.30, 0.25), 16


def _mode(t, p):
    sl = t.suggest_float(f'{p}_sl', 0.0, 40.0)
    return mp(t.suggest_int(f'{p}_slices', 2, 20), t.suggest_float(f'{p}_buy', 0.0, 10.0),
              t.suggest_float(f'{p}_tp', 0.5, 10.0), t.suggest_int(f'{p}_hold', 3, 60), sl if sl >= 3 else 0)


def build(t):
    """(engine RSI kwargs, defense, offense, breaker Policy) from a trial or ``Fixed`` params."""
    rsi = {RSI_KEYS['low']: t.suggest_float('low', 15, 45), RSI_KEYS['mid_low']: t.suggest_float('mid_low', 30, 55),
           RSI_KEYS['neutral']: t.suggest_float('neutral', 40, 60),
           RSI_KEYS['mid_high']: t.suggest_float('mid_high', 50, 75), RSI_KEYS['high']: t.suggest_float('high', 55, 85)}
    D, O = _mode(t, 'def'), _mode(t, 'off')
    k, w, h = t.suggest_int('sl_k', 1, 4), t.suggest_int('sl_win', 3, 30), t.suggest_int('sl_hold', 0, 15)
    return rsi, D, O, Policy(sl_k=k if h else 0, sl_win=w, sl_hold=h)


def seeds():
    c, _, _ = current_params()
    live = {short: c[key] for short, key in RSI_KEYS.items()}
    for p, src in (('def', 'defense'), ('off', 'offense')):
        live.update({f'{p}_slices': c[f'{src}_slices'], f'{p}_buy': c[f'{src}_buy'], f'{p}_tp': c[f'{src}_tp'],
                     f'{p}_hold': c[f'{src}_hold'], f'{p}_sl': c[f'{src}_sl']})
    return {'current': {**live, 'sl_k': 2, 'sl_win': 10, 'sl_hold': 0},
            'current + breaker': {**live, 'sl_k': 2, 'sl_win': 10, 'sl_hold': 5}}


def _window(name, prm):
    tk, start, end, real = WINDOWS[name]
    rsi, D, O, pol = build(Fixed(prm))
    kw = dict(step=1, cash=50000.0, scale=False) if real else {}
    return run(tk, rsi, D, O, start, end, policy=pol, **kw).equity


def train_stats(prm):
    use_repo_data()
    return metrics(_window('train5y', prm))


def full_stats(prm):
    use_repo_data()
    row = {}
    for name in WINDOWS:
        eq = _window(name, prm)
        row[name] = metrics(eq)
        if name == 'train5y':
            row['2022'] = eq.loc[:'2022-12-31'].iloc[-1] / eq.loc[:'2021-12-31'].iloc[-1] - 1
            row['2026-10'] = eq.iloc[-1] / eq.loc[:'2026-09-30'].iloc[-1] - 1
    return row


def objective(t):
    use_repo_data()
    rsi, D, O, pol = build(t)
    tk, start, end, _ = WINDOWS['train5y']
    return metrics(run(tk, rsi, D, O, start, end, policy=pol, step=1, cash=50000.0, scale=False).equity)


def worker(a):
    optimize(TAG, objective, a.trials, ['maximize', 'maximize'], startup=80)


def perturb(prm, rng):
    q = dict(prm)
    for k, v in prm.items():
        x = v * rng.uniform(0.9, 1.1)
        q[k] = (max(1, int(round(x))) if v else 0) if k in INT_KEYS else x
    return q


def params_text(p):
    def mode(k):
        sl = p[f'{k}_sl']
        return (f"N{p[f'{k}_slices']} buy {p[f'{k}_buy']:.2f}% tp {p[f'{k}_tp']:.2f}% hold {p[f'{k}_hold']}"
                + (f' SL {sl:.1f}%' if sl >= 3 else ' SL off'))
    rsi = ' '.join(f'{k} {p[k]:.1f}' for k in RSI_KEYS)
    brk = f"breaker {p['sl_k']} stops/{p['sl_win']}d -> defense {p['sl_hold']}d" if p['sl_hold'] else 'breaker off'
    return f'RSI {rsi} | defense {mode("def")} | offense {mode("off")} | {brk}'


def fmt(name, r):
    cells = [f'{name:<26}'] + [f'{r[w][0] * 100:6.1f}% /{r[w][1] * 100:6.1f}%' for w in WINDOWS]
    return ' | '.join(cells + [f"{r['2022'] * 100:6.1f}%", f"{r['2026-10'] * 100:6.1f}%"])


def report():
    st = load_study(TAG)
    seen, trials = set(), []
    for t in st.trials:
        if t.values is None: continue
        key = json.dumps(t.params, sort_keys=True)
        if key in seen: continue
        seen.add(key); trials.append(t)
    top = sorted([t for t in trials if t.values[1] >= -0.5], key=lambda t: -calmar(t.values))[:TOP]
    rng = random.Random(0)
    with Pool() as pool:
        again = pool.map(train_stats, [t.params for t in top])
        nb = pool.map(train_stats, [perturb(t.params, rng) for t in top for _ in range(NB)])
    cand = []
    for i, t in enumerate(top):
        n = nb[i * NB:(i + 1) * NB]
        cand.append({'params': t.params, 'stored': t.values, 'again': again[i],
                     'nb_cagr': float(np.median([x[0] for x in n])), 'nb_mdd': float(np.median([x[1] for x in n])),
                     'nb_calmar': float(np.median([calmar(x) for x in n]))})
    picks = {'pick: train Calmar': max(cand, key=lambda c: c['nb_calmar'])}
    for cap in CAPS:
        ok = [c for c in cand if c['nb_mdd'] >= -cap]
        within = [t for t in trials if t.values[1] >= -cap]
        if not ok and within:
            best = max(within, key=lambda t: t.values[0])
            ok = [{'params': best.params, 'stored': best.values, 'again': train_stats(best.params),
                   'nb_cagr': float('nan'), 'nb_mdd': float('nan'), 'nb_calmar': float('nan')}]
        if ok:
            picks[f'pick: train MDD <= {cap * 100:.0f}%'] = max(ok, key=lambda c: c['nb_cagr'])

    named = {**{n: p for n, p in seeds().items()}, **{n: c['params'] for n, c in picks.items()}}
    with Pool() as pool:
        rows = dict(zip(named, pool.map(full_stats, list(named.values()))))

    live = seeds()
    jittered = [perturb(live['current'], rng) for _ in range(NB_LIVE)]
    pairs = [({**q, 'sl_hold': 0}, {**q, **{k: live['current + breaker'][k] for k in ('sl_k', 'sl_win', 'sl_hold')}})
             for q in jittered]
    with Pool() as pool:
        nb_off = pool.map(full_stats, [a for a, _ in pairs])
        nb_on = pool.map(full_stats, [b for _, b in pairs])

    front = [t for t in st.best_trials if t.values[1] >= -0.6]
    front_rows = []
    with Pool() as pool:
        oos = pool.map(full_stats, [t.params for t in front])
    for t, r in zip(front, oos):
        front_rows.append((calmar(t.values), calmar(r['oos']), r))
    rho = pd.Series([a for a, _, _ in front_rows]).corr(pd.Series([b for _, b, _ in front_rows]), method='spearman')
    beats = sum(1 for _, _, r in front_rows if r['oos'][1] > rows['current']['oos'][1]
                and r['oos'][0] >= rows['current']['oos'][0] - 0.01)

    hdr = HEADER.replace(f"{'rule':<34}", f"{'strategy':<26}")
    out = [f'{len(trials)} unique trials; picks use the train window only; CAGR / MDD', '', hdr]
    out += [fmt(n, r) for n, r in rows.items()]
    out += ['', 'stored train value vs re-run (Fixed replay):']
    out += [f"  {n}: stored {c['stored'][0] * 100:.2f}% / {c['stored'][1] * 100:.2f}%, re-run "
            f"{c['again'][0] * 100:.2f}% / {c['again'][1] * 100:.2f}%, neighbours median "
            f"{c['nb_cagr'] * 100:.1f}% / {c['nb_mdd'] * 100:.1f}%" for n, c in picks.items()]
    out += ['', 'params:'] + [f'  {n}: {params_text(p)}' for n, p in named.items()]
    out += ['', f'Live strategy numbers jittered x U(0.9,1.1) ({NB_LIVE} draws), the same draw with and without the '
            'breaker 2/10/5: median CAGR / median MDD (worst MDD)']
    for name, nb_rows in (('current', nb_off), ('current + breaker', nb_on)):
        out.append(f'  {name:<18} ' + ' | '.join(
            f"{w} {np.median([r[w][0] for r in nb_rows]) * 100:.1f}% / {np.median([r[w][1] for r in nb_rows]) * 100:.1f}% "
            f"({min(r[w][1] for r in nb_rows) * 100:.1f}%)" for w in WINDOWS))
    for w in ('train5y', 'oos'):
        n = sum(1 for a, b in zip(nb_off, nb_on) if b[w][1] > a[w][1])
        out.append(f'  {w}: breaker has the smaller MDD in {n}/{NB_LIVE} draws')
    out += ['', f'Pareto front ({len(front_rows)} trials, train MDD >= -60%): Spearman(train Calmar, oos Calmar) '
            f'{rho:.2f}; {beats} beat current on oos MDD with CAGR >= current - 1pp', hdr]
    out += [fmt(f'front {i}', r) for i, (_, _, r) in enumerate(sorted(front_rows, key=lambda x: -x[0]))]
    text = '\n'.join(out)
    print(text)
    (RESULTS_DIR / f'{TAG}.txt').write_text(text + '\n')
    (RESULTS_DIR / f'{TAG}.json').write_text(json.dumps(
        {n: {'params': p, 'stats': rows[n]} for n, p in named.items()}, indent=1, default=float))


def main(argv):
    ap = argparse.ArgumentParser(prog='search_breaker')
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('search')
    s.add_argument('trials', type=int)
    s.add_argument('--workers', type=int, default=os.cpu_count())
    sub.add_parser('report')
    a = ap.parse_args(argv)
    if a.cmd == 'search':
        import optuna
        from benchmarks.lib.search import _storage
        st = optuna.create_study(study_name=TAG, storage=_storage(TAG), directions=['maximize', 'maximize'],
                                 load_if_exists=True)
        if not st.trials:
            for prm in seeds().values():
                st.enqueue_trial(prm)
        run_workers(worker, (a,), a.workers)
    else:
        report()
