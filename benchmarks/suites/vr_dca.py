"""적립식 VR ($5,000 + $250 every 2 weeks) vs the same deposits into buy & hold or the live strategy.

Every deposit into hold/current runs as its own scaled copy of the lump-sum curve (the engine
sizes tranches from cash, so adding cash ≈ a separate copy). Scaled prices, money ×1000.
Cell = TWR CAGR / TWR MDD | IRR | final/deposits multiple | account-value MDD.
Saved output: results/vr_dca_compare.txt
"""
from multiprocessing import Pool

import pandas as pd

from benchmarks.lib import candidates
from benchmarks.lib.data import scaled
from benchmarks.lib.engine import run
from benchmarks.lib.vr import run_vr
from benchmarks.lib.windows import metrics, windows

WIN = windows('11~all', '11~20', '21~real', 'test23~', 'syn02~')  # all run on scaled prices here
START, DEP = 5e6, 2.5e5  # 라오어 예시 $5,000 + $250/2주의 1000배


def twr(eq, flows):
    f = pd.Series(0.0, index=eq.index)
    for d, a in flows[1:]: f[d] += a
    r = (eq - f) / eq.shift(1); r.iloc[0] = 1.0
    return r.cumprod()


def xirr(eq, flows):
    t0 = eq.index[0]; T = (eq.index[-1] - t0).days / 365.25
    cf = [(-a, (d - t0).days / 365.25) for d, a in flows] + [(eq.iloc[-1], T)]
    lo, hi = -0.99, 10.0
    for _ in range(200):
        m = (lo + hi) / 2; v = sum(a / (1 + m) ** t for a, t in cf)
        lo, hi = (m, hi) if v > 0 else (lo, m)
    return m


def stat(eq, flows):
    nav = twr(eq, flows); cg, md = metrics(nav)
    dep = sum(a for _, a in flows)
    return dict(twr=cg, mdd=md, irr=xirr(eq, flows), mult=eq.iloc[-1] / dep, acct_mdd=metrics(eq)[1])


def job(a):
    kind, tgt, cfg = a; out = {}
    for name, w in WIN.items():
        t = w.ticker(tgt); s, e = w.start, w.end
        df = scaled(t).loc[s:e]
        if kind == 'vr':
            flows = []; eq = run_vr(df, START, deposit=DEP, flows=flows, **cfg)
        else:
            if kind == 'hold': base = df['Close']
            else:
                fn, D, O = candidates.current(); base = run(t, fn(t), D, O, s, e, cash=1e9).equity
            base = base / base.iloc[0]
            flows = []; run_vr(df, START, deposit=DEP, flows=flows)  # same deposit dates
            eq = sum(a * base / base.loc[d] * (base.index >= d) for d, a in flows)
        out[name] = stat(eq, flows)
    return kind, tgt, cfg, out


def main(argv):
    jobs = []
    for tgt in ('TQQQ', 'SOXL'):
        jobs += [('hold', tgt, {}), ('current', tgt, {})]
        for adv in (False, True):
            jobs.append(('vr', tgt, dict(G=10, pool_limit=0.75, advanced=adv)))
        jobs.append(('vr', tgt, dict(advanced=True, schedule=True)))
        jobs.append(('vr', tgt, dict(G=20, pool_limit=0.75, advanced=True)))
        jobs.append(('vr', tgt, dict(G=40, pool_limit=0.75, advanced=True)))
    with Pool() as p: res = p.map(job, jobs)
    print('cell = TWR CAGR / TWR MDD | IRR | 최종/원금 배수 | 계좌금액 MDD')
    for w in WIN:
        print(f'\n== {w}')
        for kind, tgt, cfg, out in res:
            o = out[w]; lab = f'{tgt} {kind} ' + ' '.join(f'{k}={v}' for k, v in cfg.items())
            print(f"{lab:46s} {o['twr']*100:6.1f}% / {o['mdd']*100:6.1f}% | IRR {o['irr']*100:6.1f}% | x{o['mult']:7.1f} | acct {o['acct_mdd']*100:6.1f}%")
