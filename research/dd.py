from evalx import *
import signals as S
c, D, O = cur_params()
idx = load('SOXL').index
def episodes(eq, k=4):
    peak = eq.cummax(); dd = eq/peak-1
    out=[]; d=dd.copy()
    for _ in range(k):
        t = d.idxmin(); v = d[t]
        if v > -0.2: break
        pk = eq.loc[:t].idxmax(); rec = eq.loc[t:][eq.loc[t:] >= eq[pk]]
        rc = rec.index[0] if len(rec) else None
        out.append((pk.date(), t.date(), rc.date() if rc is not None else None, round(v*100,1)))
        d.loc[pk:(rc or d.index[-1])] = 0
    return out
for name, ms in [('always_off', pd.Series('offense', index=idx)),
                 ('rsi_lag1', S.rsi_mode(idx, c['rsi_high_threshold'], c['rsi_mid_high'], c['rsi_neutral'], c['rsi_mid_low'], c['rsi_low_threshold'])),
                 ('sma SOXX150', S.sma_trend(idx, 'SOXX', 150))]:
    eq = run('SOXL', ms, D, O, '2011-01-01', END).equity
    print(name, episodes(eq)); print('  yearly', (yearly(eq)*100).round(0).to_dict())
eq = load('SOXL')['Close'].loc['2011':]; print('SOXL B&H yearly', (yearly(eq)*100).round(0).to_dict())
