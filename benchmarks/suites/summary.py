"""Comparison table + weekly equity curves for the web BENCHMARK tab → results/summary.json.

Rerun after strategy or engine changes, commit the JSON and redeploy; the web only reads it.
"""
import json
from datetime import datetime, timezone
from multiprocessing import Pool

import pandas as pd

from benchmarks.lib.candidates import CANDIDATES
from benchmarks.lib.data import load, scaled
from benchmarks.lib.search import RESULTS_DIR
from benchmarks.lib.windows import metrics, windows

WIN = windows('21~real', '2022', 'test23~', '11~all', 'syn02~')
WINDOW_INFO = {
    '21~real': ('2021~', '실제 SOXL, $5만, 스프레드 3주 (실제 계좌 조건)'),
    '2022': ('2022', '2022년 한 해 (핵심 하락장)'),
    'test23~': ('2023~', '실제 SOXL, $5만, 스프레드 3주'),
    '11~all': ('2011~', 'SOXL 상장 이후 전체 (가격 확대, 센트 왜곡 제거)'),
    'syn02~': ('2002~ 합성', 'SOXX 3배 합성 SOXL로 2008 폭락 포함'),
}
STRATEGIES = [  # id, name, description
    ('current', '현행 RSI', 'config/strategy.json: QQQ 주봉 RSI로 안전/공세 전환'),
    ('trend_vix', '추세 + VIX', 'trend_vix 프리셋: QQQ 30일선 ±4% 위 + VIX 26 아래면 방어, 34 위면 공세'),
    ('mix', '현행 RSI · 추세 + VIX 반반', '두 전략을 계좌를 나눠 반씩 (리밸런싱 없음)'),
    ('A', 'SOXL 160일선 + VIX', 'SOXL 160일선 ±2% 위 + VIX 38 아래면 방어, 2002~ 합성으로 학습'),
    ('B', 'SOXL 140일선 + VIX', 'SOXL 140일선 ±4% 위 + VIX 37 아래면 방어, 49 위면 공세'),
    ('C', 'SOXX 낙폭 브레이커', 'SOXX가 1년 고점 대비 -14% 아래면 방어, -5.6% 안으로 회복하면 공세'),
    ('hold', 'SOXL 보유', '매매 없이 보유 (비교 기준)'),
]


def curve(args):
    sid, w = args
    win = WIN[w]
    if sid == 'hold':
        eq = (load if win.real else scaled)(win.ticker('SOXL')).loc[win.start:win.end, 'Close']
    else:
        fn, D, O = CANDIDATES[sid]()
        eq = win.run('SOXL', fn, D, O)
    return sid, w, eq / eq.iloc[0]


def weekly(eq: pd.Series) -> pd.Series:
    """Friday closes plus the first and last day, so the chart starts at 1 and ends on the latest close."""
    w = eq.resample('W-FRI').last()
    w.index = [min(d, eq.index[-1]) for d in w.index]
    return pd.concat([eq.iloc[:1], w]).groupby(level=0).last()


def main(argv):
    ids = [sid for sid, _, _ in STRATEGIES if sid != 'mix']
    with Pool() as p: res = p.map(curve, [(sid, w) for sid in ids for w in WIN])
    curves = {(sid, w): eq for sid, w, eq in res}
    for w in WIN:
        curves[('mix', w)] = 0.5 * curves[('current', w)] + 0.5 * curves[('trend_vix', w)]
    out_windows = []
    for w, win in WIN.items():
        series = {sid: weekly(curves[(sid, w)]) for sid, _, _ in STRATEGIES}
        index = sorted(set().union(*(s.index for s in series.values())))
        stats = {}
        for sid, _, _ in STRATEGIES:
            cagr, mdd = win.stat(curves[(sid, w)])
            stats[sid] = {'cagr': round(float(cagr), 4), 'mdd': round(float(mdd), 4)}
        out_windows.append({
            'id': w, 'label': WINDOW_INFO[w][0], 'description': WINDOW_INFO[w][1],
            'start': index[0].date().isoformat(), 'end': index[-1].date().isoformat(),
            'total_return': win.total_return,
            'dates': [d.date().isoformat() for d in index],
            'curves': {sid: [round(float(v), 4) for v in s.reindex(index).ffill()] for sid, s in series.items()},
            'stats': stats,
        })
    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'data_end': load('SOXL').index[-1].date().isoformat(),
        'strategies': [{'id': sid, 'name': n, 'description': d} for sid, n, d in STRATEGIES],
        'windows': out_windows,
    }
    path = RESULTS_DIR / 'summary.json'
    path.write_text(json.dumps(summary, ensure_ascii=False, separators=(',', ':')))
    print(f'{path} ({path.stat().st_size / 1024:.0f} KB)')
    for wd in out_windows:
        print(f"\n== {wd['label']}")
        for sid, name, _ in STRATEGIES:
            st = wd['stats'][sid]
            print(f"  {name:20s} {st['cagr']*100:7.1f}% / {st['mdd']*100:6.1f}%")
