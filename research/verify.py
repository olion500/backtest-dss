"""Harness sanity check: the injected lag-1 RSI series must reproduce the engine's own RSI path."""
from evalx import *
from signals import rsi_mode
c, D, O = cur_params()
kw = dict(rsi_high_threshold=c['rsi_high_threshold'], rsi_mid_high=c['rsi_mid_high'], rsi_neutral=c['rsi_neutral'],
          rsi_mid_low=c['rsi_mid_low'], rsi_low_threshold=c['rsi_low_threshold'])
idx = load('SOXL').index
ms = rsi_mode(idx, c['rsi_high_threshold'], c['rsi_mid_high'], c['rsi_neutral'], c['rsi_mid_low'], c['rsi_low_threshold'])
for s, e in [('2011-01-01', END), ('2021-01-01', END), ('2022-01-01', '2022-12-31')]:
    a = metrics(run('SOXL', kw, D, O, s, e).equity); b = metrics(run('SOXL', ms, D, O, s, e).equity)
    print(s, 'engine', ['%.4f' % x for x in a], 'injected', ['%.4f' % x for x in b], 'OK' if a == b else 'MISMATCH')
