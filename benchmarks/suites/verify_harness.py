"""Check: the injected lag-1 RSI series reproduces the engine's own RSI backtest exactly.

Run this after any engine change; every other suite relies on the injection matching the engine.
"""
from benchmarks.lib.data import load
from benchmarks.lib.engine import current_params, run
from benchmarks.lib.signals import rsi_mode
from benchmarks.lib.windows import END, metrics


def main(argv):
    c, D, O = current_params()
    kw = dict(rsi_high_threshold=c['rsi_high_threshold'], rsi_mid_high=c['rsi_mid_high'], rsi_neutral=c['rsi_neutral'],
              rsi_mid_low=c['rsi_mid_low'], rsi_low_threshold=c['rsi_low_threshold'])
    ms = rsi_mode(load('SOXL').index, c['rsi_high_threshold'], c['rsi_mid_high'], c['rsi_neutral'],
                  c['rsi_mid_low'], c['rsi_low_threshold'])
    for s, e in [('2011-01-01', END), ('2021-01-01', END), ('2022-01-01', '2022-12-31')]:
        a = metrics(run('SOXL', kw, D, O, s, e).equity)
        b = metrics(run('SOXL', ms, D, O, s, e).equity)
        print(s, 'engine', ['%.4f' % x for x in a], 'injected', ['%.4f' % x for x in b], 'OK' if a == b else 'MISMATCH')
