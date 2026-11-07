"""'일본 할아버지' rule-based scaling simulator for comparisons.

Moves are measured from the cycle's basis price (average cost by default) at the close:
    down 10% -> buy 10% of the cycle budget, down 25% -> buy 25% more (5% down: nothing)
    up 25/35/45/60% -> sell 10/20/30/40% of the shares held, up 100% -> sell everything
Each level fires once per cycle. A cycle starts with ``init_frac`` of equity in stock (65% so
the two buys bring it to 100%); after a full sale the next cycle starts at the next close.
Fills are at the close (LOC-style), integer shares, no costs.

basis='avg': average cost (수익률), moves after each buy. basis='entry': the cycle's first fill.
basis='market': the market's own moves. Drops from the highest close (the buys re-arm at every new
high); rises from the lowest close since the latest drop episode began (a drop buy firing starts
one and re-arms the sells), or since the cycle start before any.
basis='mixed': drops from the market high (as 'market'), rises from the average cost (as 'avg').
"""
import math

import numpy as np
import pandas as pd

BUYS = ((0.10, 0.10), (0.25, 0.25))  # (drop, fraction of the cycle budget)
SELLS = ((0.25, 0.10), (0.35, 0.20), (0.45, 0.30), (0.60, 0.40), (1.00, 1.0))  # (rise, fraction held)


def run_grandpa(df, cash, init_frac=0.65, basis='avg', buys=BUYS, sells=SELLS, stats=None):
    """Equity curve. stats: optional dict that receives cycles, average stock weight and level hits."""
    c = df['Close'].to_numpy()
    q, cost, entry = 0, 0.0, 0.0
    budget, fired, peak, trough = 0.0, set(), 0.0, 0.0
    market_down, market_up = basis in ('market', 'mixed'), basis == 'market'
    eq, weight = np.empty(len(c)), np.empty(len(c))
    hits = {('buy', d): 0 for d, _ in buys} | {('sell', u): 0 for u, _ in sells}
    cycles = 0
    for i, px in enumerate(c):
        if q == 0:  # new cycle
            budget, fired, peak, trough = cash, set(), px, px
            n = math.floor(budget * init_frac / px)
            q, cost, entry = n, n * px, px
            cash -= n * px
            cycles += 1
        else:
            if px > peak:
                peak = px
                if market_down: fired -= {('buy', d) for d, _ in buys}
            trough = min(trough, px)
            ref = cost / q if basis in ('avg', 'mixed') else entry
            down = px / peak - 1 if market_down else px / ref - 1
            for d, f in buys:
                if ('buy', d) not in fired and down <= -d:
                    if market_up and not any(k[0] == 'buy' for k in fired):  # a new drop episode
                        trough = px; fired -= {('sell', u) for u, _ in sells}
                    fired.add(('buy', d)); hits['buy', d] += 1
                    n = math.floor(min(budget * f, cash) / px)
                    q += n; cost += n * px; cash -= n * px
            up = px / trough - 1 if market_up else px / (cost / q if basis in ('avg', 'mixed') else entry) - 1
            for u, f in sells:
                if ('sell', u) not in fired and up >= u:
                    fired.add(('sell', u)); hits['sell', u] += 1
                    n = q if f >= 1 else math.floor(q * f)
                    cost -= cost / q * n if q else 0.0
                    q -= n; cash += n * px
        eq[i] = cash + q * px
        weight[i] = q * px / eq[i]
    if stats is not None:
        stats.update(cycles=cycles, weight=float(weight.mean()), hits=hits)
    return pd.Series(eq, index=df.index)
