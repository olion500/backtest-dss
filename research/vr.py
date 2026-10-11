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


def run_vr(df, cash, stock_frac=0.5, G=10, band=0.15, pool_limit=0.5, advanced=True, fill='close',
           weeks=2, deposit=0.0, schedule=False, flows=None):
    """deposit: added to both pool and V every cycle (적립식). schedule=True uses the simulator's
    default ramps: G = 10 + years, pool limit 75% falling 5%p every half year to 10%.
    flows: optional list that receives (date, amount) for every deposit, initial cash included."""
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
            V = max(V + P / g + ((E - V) / (2 * math.sqrt(g)) if advanced else 0.0) + deposit, 0.0)
            P += deposit
            if deposit and flows is not None: flows.append((idx[i], deposit))
            usable, saved = P * lim, P * (1 - lim)
        top, bot = V * (1 + band), V * (1 - band)
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
