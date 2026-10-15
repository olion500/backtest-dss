"""Check: the engine's native trend_vix mode matches the research sma_vix signal.

Labels are swapped: research 'offense' (strong market) = engine 'defense'.
"""
import json

import engines.dongpa_engine as E
from engines.dongpa_engine import CapitalParams, StrategyParams

from benchmarks.lib.data import load, scaled
from benchmarks.lib.engine import run
from benchmarks.lib.search import RESULTS_DIR, Fixed, build
from benchmarks.lib.windows import END, metrics


def main(argv):
    p = json.loads((RESULTS_DIR / 'seed_sma_vix0.json').read_text())
    ms, D_r, O_r = build(Fixed(p), 'sma_vix', load('SOXL').index)
    native = dict(mode_switch_strategy='trend_vix', trend_sma_days=p['n'], trend_band_pct=p['band'] * 100,
                  vix_low=p['lo'], vix_high=p['lo'] + p['hi_gap'])
    for s, e, step, cash, sc in (('2011-01-01', END, None, 1e9, True), ('2021-01-01', END, 3, 50000.0, False),
                                 ('2022-01-01', '2022-12-31', 3, 50000.0, False)):
        a = metrics(run('SOXL', ms, D_r, O_r, s, e, step=step, cash=cash, scale=sc).equity)
        t = (scaled('SOXL') if sc else load('SOXL')).loc[s:e]
        params = StrategyParams(target_ticker='SOXL', momentum_ticker='QQQ', defense=O_r, offense=D_r,
                                spread_buy_step=step, **native)
        r = E.run_backtest(t, load('QQQ').loc[:e], params, CapitalParams(initial_cash=cash), vix_data=load('VIX').loc[:e])
        b = metrics(r.equity)
        ok = max(abs(x - y) for x, y in zip(a, b)) < 1e-9
        print(s, 'research', ['%.4f' % x for x in a], 'engine', ['%.4f' % x for x in b], 'OK' if ok else 'MISMATCH')
