"""Value Rebalancing (라오어 VR 5.0), lump-sum (거치식) version.

Every 2 weeks the target value V moves:
    basic:    V' = V + Pool/G
    advanced: V' = V + Pool/G + (E - V) / (2*sqrt(G))      (실력공식, E = stock value)
Band = V*(1-b) .. V*(1+b). Above the ceiling, sell back to the ceiling; below the floor, buy back to
the floor using at most `pool_limit` of the pool as it stood at the cycle start. Sale proceeds go
to the pool but are not spendable until the next cycle (same as droomii/vr-simul).

fill='close': orders fill at the close (LOC-style, like the simulator and our engine).
fill='limit': the real buy/sell tables, one share per level at floor/(Q+k) and ceiling/(Q-k),
              filled intraday when Low/High reaches them (at the open if it gaps through).
The cycle update uses the previous close, so nothing looks ahead.
"""
import math
import numpy as np, pandas as pd

# 상승률 표 (cafe.naver.com/infinitebuying/27118, 사용자 구글 시트 '상승률' 시트): P/V 하한 → (평가금<V, 평가금>V)
RATE_TABLE = [(1.1, 1.11, 1.115), (1.05, 1.105, 1.11), (1.0, 1.1, 1.105), (0.95, 1.095, 1.1), (0.9, 1.09, 1.095),
              (0.85, 1.085, 1.09), (0.8, 1.08, 1.085), (0.75, 1.075, 1.08), (0.7, 1.07, 1.075), (0.65, 1.065, 1.07),
              (0.6, 1.06, 1.065), (0.55, 1.055, 1.06), (0.5, 1.05, 1.055), (0.45, 1.045, 1.05), (0.4, 1.04, 1.045),
              (0.35, 1.035, 1.04), (0.3, 1.03, 1.035), (0.25, 1.025, 1.03), (0.2, 1.02, 1.025), (0.15, 1.015, 1.02),
              (0.1, 1.01, 1.015), (0.05, 1.005, 1.01), (0.01, 1.001, 1.005), (0.0, 1.0, 1.001)]


def table_rate(pv, above):
    for lo, under_v, over_v in RATE_TABLE:
        if pv >= lo: return over_v if above else under_v
    return RATE_TABLE[-1][2 if above else 1]


def run_vr(df, cash, stock_frac=0.5, G=10, band=0.15, pool_limit=0.5, advanced=True, fill='close',
           weeks=2, deposit=0.0, schedule=False, flows=None, table=False):
    """deposit: added to both pool and V every cycle (적립식). schedule=True uses the simulator's
    default ramps: G = 10 + years, pool limit 75% falling 5%p every half year to 10%.
    flows: optional list that receives (date, amount) for every deposit, initial cash included.
    table=True: the user's sheet rule instead of P/G — V' = V × 상승률표(P/V, 평가금>V) + deposit,
    band -20% / +25% (band arg ignored)."""
    c, lo, hi, op = (df[k].to_numpy() for k in ('Close', 'Low', 'High', 'Open'))
    idx = df.index
    cyc = ((idx - pd.Timestamp('2000-01-03')).days // 7 // weeks).to_numpy()
    q = math.floor(cash * stock_frac / c[0])
    pool = cash - q * c[0]
    V = q * c[0]
    usable, saved = pool * pool_limit, pool * (1 - pool_limit)
    eq = np.empty(len(df)); eq[0] = pool + q * c[0]
    if flows is not None: flows.append((idx[0], cash))
    ncyc = 0
    for i in range(1, len(df)):
        if cyc[i] != cyc[i - 1]:
            ncyc += 1; wk = ncyc * weeks
            g = 10 + wk // 52 if schedule else G
            lim = 1 - min(0.25 + wk // 26 * 0.05, 0.9) if schedule else pool_limit
            E = q * c[i - 1]; P = usable + saved
            if table:
                V = V * table_rate(P / V if V > 0 else 0.0, E > V) + deposit
            else:
                V = max(V + P / g + ((E - V) / (2 * math.sqrt(g)) if advanced else 0.0) + deposit, 0.0)
            P += deposit
            if deposit and flows is not None: flows.append((idx[i], deposit))
            usable, saved = P * lim, P * (1 - lim)
        top, bot = (V * 1.25, V * 0.8) if table else (V * (1 + band), V * (1 - band))
        if fill == 'close':
            mv = q * c[i]
            if mv > top:
                n = math.floor((mv - top) / c[i]); q -= n; saved += n * c[i]
            elif mv < bot:
                n = math.floor(min(bot - mv, usable) / c[i]); q += n; usable -= n * c[i]
        else:
            q0 = q
            k = 1  # sell table: level k at top/(q0-k), ascending
            while k < q0 and top / (q0 - k) <= hi[i]:
                px = max(top / (q0 - k), op[i]); q -= 1; saved += px; k += 1
            k = 1  # buy table: level k at bot/(q0+k), descending
            while True:
                px = bot / (q0 + k)
                if px < lo[i]: break
                px = min(px, op[i])
                if px > usable: break
                q += 1; usable -= px; k += 1
        eq[i] = usable + saved + q * c[i]
    return pd.Series(eq, index=idx)
