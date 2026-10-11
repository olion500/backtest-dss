"""적립식 comparison: VR with per-cycle deposits vs putting the same deposits into buy & hold or the
current strategy. Every deposit into hold/current runs as its own scaled copy of the lump-sum curve
(the engine sizes tranches from cash, so adding cash ≈ a separate copy). Scaled prices, money ×1000."""
from multiprocessing import Pool
from harness import *
from vr import run_vr
import candidates
END = '2026-10-09'
WIN = {'11~all': ('2011-01-01', END), '11~20': ('2011-01-01', '2020-12-31'),
       '21~': ('2021-01-01', END), '23~': ('2023-01-01', END), '02~synth': ('2002-01-01', END)}
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
    for w, (s, e) in WIN.items():
        t = tgt + ('_X' if w == '02~synth' else ''); df = scaled(t).loc[s:e]
        if kind == 'vr':
            flows = []; eq = run_vr(df, START, deposit=DEP, flows=flows, **cfg)
        else:
            if kind == 'hold': base = df['Close']
            else:
                fn, D, O = candidates.current(); base = run(t, fn(t), D, O, s, e, cash=1e9).equity
            base = base / base.iloc[0]
            flows = []; run_vr(df, START, deposit=DEP, flows=flows)  # same deposit dates
            eq = sum(a * base / base.loc[d] * (base.index >= d) for d, a in flows)
        out[w] = stat(eq, flows)
    return kind, tgt, cfg, out

if __name__ == '__main__':
    jobs = []
    for tgt in ('TQQQ', 'SOXL'):
        jobs += [('hold', tgt, {}), ('current', tgt, {})]
        for adv in (False, True):
            jobs.append(('vr', tgt, dict(G=10, pool_limit=0.75, advanced=adv)))
        jobs.append(('vr', tgt, dict(advanced=True, schedule=True)))
        jobs.append(('vr', tgt, dict(G=20, pool_limit=0.75, advanced=True)))
        jobs.append(('vr', tgt, dict(G=40, pool_limit=0.75, advanced=True)))
    with Pool(12) as p: res = p.map(job, jobs)
    print('cell = TWR CAGR / TWR MDD | IRR | 최종/원금 배수 | 계좌금액 MDD')
    for w in WIN:
        print(f'\n== {w}')
        for kind, tgt, cfg, out in res:
            o = out[w]; lab = f'{tgt} {kind} ' + ' '.join(f'{k}={v}' for k, v in cfg.items())
            print(f"{lab:46s} {o['twr']*100:6.1f}% / {o['mdd']*100:6.1f}% | IRR {o['irr']*100:6.1f}% | x{o['mult']:7.1f} | acct {o['acct_mdd']*100:6.1f}%")
