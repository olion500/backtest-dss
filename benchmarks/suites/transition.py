"""공세→안전 전환 손실 대책(전환 시 공세 로트 처리, 조기 안전)을 최근 5년 학습 → 10·15년·이전 10년 시험.

usage: transition [--workers N]   → results/transition.txt
Rules: lib/transition.py (Policy). The live strategy (config/strategy.json, engine RSI) is the base.
Train = last 5 years (2021-10-25~) like the accounts (real prices, $50k, 1-share spread steps); the
pick is the best train Calmar (CAGR / |MDD|). Test: 10y (2016-10-25~) and 15y (2011-10-25~) hold
the train years, so the clean test is oos (2011-10-25~2021-10-24, never seen); syn = synthetic
SOXL_X 2002~2010 (2008). Long windows use scaled prices and $1e9 like search_mdd.
Every rule is run on every window; only the train numbers choose. The top picks are re-run with
their numbers jittered ×U(0.9, 1.1) (robust section).
2026-11-01 run: transition --workers 12
"""
import argparse
import random
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from benchmarks.lib.data import use_repo_data
from benchmarks.lib.engine import current_params
from benchmarks.lib.search import RESULTS_DIR
from benchmarks.lib.transition import KEEP, Policy, run
from benchmarks.lib.windows import metrics

WINDOWS = {
    'train5y': ('SOXL', '2021-10-25', None, True),
    '10y': ('SOXL', '2016-10-25', None, False),
    '15y': ('SOXL', '2011-10-25', None, False),
    'oos': ('SOXL', '2011-10-25', '2021-10-24', False),
    'syn': ('SOXL_X', '2002-01-01', '2010-12-31', False),
}
SWITCH = [Policy('keep'), Policy('liquidate'), Policy('losers'), Policy('retarget')] + [
    Policy('slcap', sl_cap=x) for x in (5, 8, 10, 12, 15)]
EARLY_DD = [Policy(dd_n=n, dd_pct=p, dd_hold=h) for n in (5, 10, 20, 40) for p in (10, 15, 20, 25, 30)
            for h in (0, 5, 10, 20)]
EARLY_SL = [Policy(sl_k=k, sl_win=w, sl_hold=h) for k in (1, 2, 3) for w in (5, 10, 20) for h in (5, 10, 20, 40)]
TOP_COMBO, TOP_ROBUST, NB = 5, 3, 12
SL_FAMILY = [Policy(s, x, sl_k=k, sl_win=w, sl_hold=h) for s, x in (('keep', 0.0), ('slcap', 10.0))
             for k in (1, 2, 3) for w in (5, 10, 20) for h in (5, 10)]
RECOMMENDED = [Policy(sl_k=2, sl_win=10, sl_hold=5), Policy('slcap', 10.0, sl_k=2, sl_win=20, sl_hold=5)]
PAUSE = ([Policy(pause_pre=True, pause_both=b) for b in (False, True)]
         + [Policy(pause_post=n, pause_both=b) for n in (3, 5, 10, 20) for b in (False, True)]
         + [Policy(pause_pre=True, pause_post=n) for n in (3, 5, 10)]
         + [Policy(sl_k=k, sl_win=10, sl_hold=h, action='pause') for k in (1, 2, 3) for h in (3, 5, 10, 20)]
         + [Policy(dd_n=n, dd_pct=p, dd_hold=h, action='pause') for n in (10, 20) for p in (20, 25, 30)
            for h in (0, 5, 10)]
         + [Policy('liquidate', pause_post=n) for n in (5, 10, 20)]
         + [Policy('liquidate', pause_pre=True, pause_post=5)]
         + [Policy(sl_k=2, sl_win=10, sl_hold=5, pause_pre=True), Policy(sl_k=2, sl_win=10, sl_hold=5, pause_post=5)])


def _base():
    c, D, O = current_params()
    rsi = {k: c[k] for k in ('rsi_high_threshold', 'rsi_mid_high', 'rsi_neutral', 'rsi_mid_low', 'rsi_low_threshold')}
    return rsi, D, O


def evaluate(policy: Policy) -> dict:
    use_repo_data()
    rsi, D, O = _base()
    row = {'policy': policy}
    for name, (tk, start, end, real) in WINDOWS.items():
        kw = dict(step=1, cash=50000.0, scale=False) if real else {}
        eq = run(tk, rsi, D, O, start, end, policy=policy, **kw).equity
        row[name] = metrics(eq)
        if name == 'train5y':
            row['2022'] = eq.loc[:'2022-12-31'].iloc[-1] / eq.loc[:'2021-12-31'].iloc[-1] - 1
            row['2026-10'] = eq.iloc[-1] / eq.loc[:'2026-09-30'].iloc[-1] - 1
            row['last'] = eq.index[-1].date()
    return row


def calmar(cm):
    return cm[0] / abs(cm[1]) if cm[1] else float('inf')


def run_all(policies, workers):
    with Pool(workers) as pool:
        return pool.map(evaluate, policies)


def jitter(p: Policy, rng) -> Policy:
    f = lambda v, lo=1: max(lo, int(round(v * rng.uniform(0.9, 1.1)))) if v else v
    g = lambda v: v * rng.uniform(0.9, 1.1) if v else v
    return replace(p, sl_cap=g(p.sl_cap), dd_n=f(p.dd_n), dd_pct=g(p.dd_pct), dd_hold=f(p.dd_hold, 0),
                   sl_win=f(p.sl_win), sl_hold=f(p.sl_hold), pause_post=f(p.pause_post))


def fmt(r):
    cells = [f"{r['policy'].name:<34}"]
    for w in WINDOWS:
        c, m = r[w]
        cells.append(f'{c * 100:6.1f}% /{m * 100:6.1f}%')
    cells.append(f"{r['2022'] * 100:6.1f}%")
    cells.append(f"{r['2026-10'] * 100:6.1f}%")
    return ' | '.join(cells)


HEADER = ' | '.join([f"{'rule':<34}"] + [f'{w:^16}' for w in WINDOWS] + ['  2022 ', '2026-10'])


def main(argv):
    ap = argparse.ArgumentParser(prog='transition')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args(argv)

    single = run_all(SWITCH + EARLY_DD + EARLY_SL, a.workers)
    early = sorted([r for r in single if r['policy'].on_switch == 'keep' and r['policy'] != KEEP],
                   key=lambda r: -calmar(r['train5y']))[:TOP_COMBO]
    combos = [Policy(s.on_switch, s.sl_cap, e['policy'].dd_n, e['policy'].dd_pct, e['policy'].dd_hold,
                     e['policy'].sl_k, e['policy'].sl_win, e['policy'].sl_hold)
              for e in early for s in SWITCH[1:]]
    dd_top = [r['policy'] for r in early if r['policy'].dd_n][:3]
    sl_top = [r['policy'] for r in sorted(single, key=lambda r: -calmar(r['train5y'])) if r['policy'].sl_k][:3]
    combos += [Policy(dd_n=d.dd_n, dd_pct=d.dd_pct, dd_hold=d.dd_hold, sl_k=s.sl_k, sl_win=s.sl_win,
                      sl_hold=s.sl_hold) for d in dd_top for s in sl_top]
    combos += [p for p in SL_FAMILY + RECOMMENDED + PAUSE if p not in set(combos) | {r['policy'] for r in single}]
    rows = single + run_all(combos, a.workers)
    base = next(r for r in rows if r['policy'] == KEEP)
    by_train = sorted(rows, key=lambda r: -calmar(r['train5y']))
    best_pause = max((r for p in PAUSE for r in rows if r['policy'] == p), key=lambda r: calmar(r['train5y']))

    tr = np.array([calmar(r['train5y']) for r in rows])
    rank_corr = {w: pd.Series(tr).corr(pd.Series([calmar(r[w]) for r in rows]), method='spearman')
                 for w in WINDOWS if w != 'train5y'}

    rng = random.Random(0)
    robust = []
    for r in by_train[:TOP_ROBUST] + [r for r in rows if r['policy'] in RECOMMENDED] + [best_pause]:
        nb = run_all([jitter(r['policy'], rng) for _ in range(NB)], a.workers)
        med = {w: tuple(np.median([n[w][k] for n in nb]) for k in (0, 1)) for w in WINDOWS}
        worst = {w: min(n[w][1] for n in nb) for w in WINDOWS}
        robust.append((r, med, worst))

    out = [f"data through {base['last']}; CAGR / MDD; 2022 = return; 2026-10 = since 2026-09-30 (train run)", '',
           '## Current (keep) and the switch rules alone', HEADER]
    out += [fmt(r) for r in rows if r['policy'] in SWITCH]
    out += ['', f'## Top 15 by train Calmar (of {len(rows)} rules; pick = first row)', HEADER]
    out += [fmt(r) for r in by_train[:15]]
    out += ['', '## Best train Calmar per family', HEADER]
    fams = {'switch only': lambda p: p.on_switch != 'keep' and not p.dd_n and not p.sl_k,
            'dd only': lambda p: p.on_switch == 'keep' and p.dd_n and not p.sl_k,
            'sl streak only': lambda p: p.on_switch == 'keep' and p.sl_k and not p.dd_n}
    for name, pred in fams.items():
        cand = [r for r in by_train if pred(r['policy'])]
        if cand: out.append(fmt(cand[0]) + f'   <- {name}')
    out += ['', '## Offense stop-loss streak breaker (sl_k stops within sl_win sessions -> defense sl_hold)', HEADER]
    out += [fmt(r) for p in SL_FAMILY for r in rows if r['policy'] == p]
    for w in WINDOWS:
        n = sum(1 for p in SL_FAMILY for r in rows if r['policy'] == p and r[w][1] > base[w][1]
                and r[w][0] >= base[w][0] - 0.01)
        out.append(f'{w}: {n}/{len(SL_FAMILY)} breaker rules beat keep on MDD with CAGR >= keep - 1pp')
    out += ['', '## No new buys around the switch (pause; open lots still sell) vs keep and the recommended breaker',
            HEADER, fmt(base) + '   <- keep', fmt(next(r for r in rows if r['policy'] == RECOMMENDED[0])) + '   <- recommended']
    out += [fmt(r) for p in PAUSE for r in rows if r['policy'] == p]
    for w in WINDOWS:
        n = sum(1 for p in PAUSE for r in rows if r['policy'] == p and r[w][1] > base[w][1]
                and r[w][0] >= base[w][0] - 0.01)
        out.append(f'{w}: {n}/{len(PAUSE)} pause rules beat keep on MDD with CAGR >= keep - 1pp')
    out.append(fmt(best_pause) + '   <- best pause rule by train Calmar')
    out += ['', 'Spearman(train Calmar, test Calmar) over all rules: ' +
            ', '.join(f'{w} {v:.2f}' for w, v in rank_corr.items()), '',
            f'## Robust: top {TOP_ROBUST} train picks + recommended, {NB} neighbours ×U(0.9,1.1): median CAGR / MDD, worst MDD']
    for r, med, worst in robust:
        out.append(f"{r['policy'].name}")
        out.append('   ' + ' | '.join(f'{w} {med[w][0] * 100:.1f}% / {med[w][1] * 100:.1f}% (worst {worst[w] * 100:.1f}%)'
                                      for w in WINDOWS))
    text = '\n'.join(out)
    print(text)
    (RESULTS_DIR / 'transition.txt').write_text(text + '\n')
