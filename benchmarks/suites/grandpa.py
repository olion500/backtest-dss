"""'일본 할아버지' 분할 매수·매도 규칙 (lib/grandpa.py) vs 현행 vs 보유, SOXL·TQQQ·QQQ·SPY.

Saved output: results/grandpa_compare.txt
"""
from multiprocessing import Pool

from benchmarks.lib import candidates
from benchmarks.lib.data import load, scaled
from benchmarks.lib.grandpa import run_grandpa
from benchmarks.lib.windows import windows

WIN = windows('11~all', '11~20', '21~real', '2022', 'test23~', 'syn02~')
LEVERAGED = ('SOXL', 'TQQQ')  # only these have synthetic 2002~ series and the live strategy


def job(a):
    kind, tgt, cfg = a; out = {}; st = {}
    for name, w in WIN.items():
        t = w.ticker(tgt) if tgt in LEVERAGED else tgt
        df = (load if w.real else scaled)(t).loc[w.start:w.end]
        if kind == 'current':
            fn, D, O = candidates.current()
            eq = w.run(tgt, fn, D, O)
        elif kind == 'hold':
            eq = df['Close']
        else:
            s = {}
            eq = run_grandpa(df, w.cash, stats=s, **cfg)
            st[name] = s
        out[name] = w.stat(eq / eq.iloc[0])
    return kind, tgt, cfg, out, st


def main(argv):
    jobs = []
    for tgt in ('SOXL', 'TQQQ', 'QQQ', 'SPY'):
        if tgt in LEVERAGED:
            jobs.append(('current', tgt, {}))
        jobs.append(('hold', tgt, {}))
        for basis in ('avg', 'entry', 'market', 'mixed'):
            for init in (0.65, 0.5, 1.0):
                jobs.append(('grandpa', tgt, dict(basis=basis, init_frac=init)))
    with Pool() as p: res = p.map(job, jobs)
    print('CAGR / MDD (2022: 연간 수익률).  [w] 평균 주식 비중, [c] 사이클 수 (11~all)')
    print(f"{'':40s}" + ''.join(f'{w:>17s}' for w in WIN) + '      w     c')
    for kind, tgt, cfg, out, st in res:
        lab = f'{tgt} {kind} ' + ' '.join(f'{k}={v}' for k, v in cfg.items())
        tail = f"  {st['11~all']['weight']*100:4.0f}% {st['11~all']['cycles']:5d}" if st else ''
        print(f'{lab:40s}' + ''.join(f'{out[w][0]*100:7.1f}/{out[w][1]*100:6.1f}%  ' for w in WIN) + tail)
    print('\n레벨 도달 횟수 (11~all, init=0.65)')
    for kind, tgt, cfg, out, st in res:
        if kind == 'grandpa' and cfg['init_frac'] == 0.65 and cfg['basis'] in ('avg', 'market'):
            h = st['11~all']['hits']
            print(f"{tgt:5s} {cfg['basis']:6s} " + '  '.join(f"{k[0]}{'-' if k[0] == 'buy' else '+'}{k[1]*100:.0f}%:{v}" for k, v in h.items()))
