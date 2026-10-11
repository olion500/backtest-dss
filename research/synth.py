"""Synthetic leveraged ETFs before listing: lev × base daily return − (expense + (lev−1) × T-bill) / 252.
`er` includes an extra drag calibrated so the overlap with the real fund matches (~0%/yr gap)."""
import numpy as np, pandas as pd
from harness import load, DL


def synth(base, real, lev=3, er=0.0095, out=None):
    b = load(base)
    rf = load('IRX')['Close'].reindex(b.index).ffill().fillna(0) / 100
    r = (lev * b['Adj Close'].pct_change() - (er + (lev - 1) * rf) / 252).fillna(0)

    rl = load(real)['Adj Close'].pct_change()
    ov = rl.dropna().index.intersection(r.index)[5:]
    a, s = (1 + rl[ov]).prod(), (1 + r[ov]).prod()
    print(f'{base}x{lev} vs {real}: corr {np.corrcoef(rl[ov], r[ov])[0, 1]:.4f}  real total x{a:.1f}  synth x{s:.1f}  '
          f'ann diff {((s / a) ** (252 / len(ov)) - 1) * 100:+.2f}%/yr')
    for y in (2011, 2015, 2018, 2020, 2022, 2024):
        m = ov[ov.year == y]
        print(f'   {y}: real {((1 + rl[m]).prod() - 1) * 100:6.1f}%  synth {((1 + r[m]).prod() - 1) * 100:6.1f}%')

    # Splice: back-cast closes before the real fund's first day so that close[t+1]/close[t] = 1 + r[t+1].
    first = load(real).index[0]
    pre = r[r.index < first]
    c_real0 = load(real)['Close'].iloc[0]
    growth_after = np.r_[(1 + pre.values[::-1]).cumprod()[::-1][1:], 1.0] * (1 + r.get(first, 0))
    pre_close = pd.Series(c_real0 / growth_after, index=pre.index)
    d = pd.DataFrame({'Open': pre_close, 'High': pre_close, 'Low': pre_close, 'Close': pre_close,
                      'Adj Close': pre_close, 'Volume': 0})
    full = pd.concat([d, load(real)[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]])
    if out:
        full.index.name = 'Date'
        full.to_csv(f'{DL}/{out}.csv')
    return full


if __name__ == '__main__':
    synth('SOXX', 'SOXL', er=0.0095 + 0.014, out='SOXL_X')
    synth('QQQ', 'TQQQ', er=0.0095 + 0.0104, out='TQQQ_X')
    synth('SOXX', 'USD', lev=2, er=0.0095, out='USD_X')
