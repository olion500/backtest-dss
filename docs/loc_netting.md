# LOC Order Netting Logic

This document explains the netting (퉁치기) algorithm for LOC (Limit-On-Close) orders.

## Basic Principle

LOC orders execute at the market close price. Netting can only occur when buy and sell orders have **overlapping execution ranges**.

| Condition | Result |
|-----------|--------|
| Sell Price ≤ Buy Price | Overlapping range exists → Netting applies |
| Sell Price > Buy Price | No overlapping range → No netting |

### LOC Order Execution Rules

- **Buy LOC at price Y**: Executes if `close price ≤ Y`
- **Sell LOC at price X**: Executes if `close price ≥ X`

For both orders to execute simultaneously, we need: `X ≤ close ≤ Y`

This is only possible when `X ≤ Y` (sell price ≤ buy price).

## Cases

### Case 1: Sell Price < Buy Price (Netting Occurs)

```
Buy at $100 for 500 shares
Sell at $98 for 300 shares

Price ranges:
- Close < $98: Only buy executes (500 shares)
- $98 ≤ Close ≤ $100: BOTH execute → Net buy 200 shares
- Close > $100: Only sell executes (300 shares)
```

When the close price falls in the overlapping range ($98-$100), both orders execute at the same price. The broker can net them to deliver only 200 shares (net buy).

### Case 2: Sell Price > Buy Price (No Netting)

```
Buy at $100 for 500 shares
Sell at $105 for 300 shares

Price ranges:
- Close ≤ $100: Only buy executes (500 shares)
- $100 < Close < $105: Neither executes
- Close ≥ $105: Only sell executes (300 shares)
```

There is no close price where both orders can execute simultaneously, so netting is impossible.

### Case 3: Sell Price = Buy Price (Netting at Exact Price)

```
Buy at $100 for 500 shares
Sell at $100 for 300 shares

Price ranges:
- Close < $100: Only buy executes (500 shares)
- Close = $100: Both execute → Net buy 200 shares
- Close > $100: Only sell executes (300 shares)
```

### Case 4: Multiple Sell Orders (Cumulative Sum)

```
Buy at $100 for 500 shares
Sell at $95 for 100 shares
Sell at $97 for 150 shares
Sell at $98 for 200 shares
```

**Price-range analysis using cumulative sum:**

Sort sells by price ascending, accumulate sell qty, compute `net = buy_qty - cum_sell` at each boundary:

| Close Range | Cum Sell | Buy | Net = Buy - Cum Sell | Result |
|-------------|---------|-----|---------------------|--------|
| < $95 | 0 | 500 | **+500** | 매수 (전량) |
| $95 ~ $97 | 100 | 500 | **+400** | 순매수 |
| $97 ~ $98 | 250 | 500 | **+250** | 순매수 |
| $98 ~ $100 | 450 | 500 | **+50** | 순매수 (퉁치기 결과) |
| > $100 | 450 | 0 | **-450** | 매도 (전량) |

### Case 5: Multiple Sells, Sell Side Larger

```
Buy at $100 for 200 shares
Sell at $95 for 100 shares
Sell at $97 for 150 shares
Sell at $98 for 200 shares
```

| Close Range | Cum Sell | Buy | Net | Result |
|-------------|---------|-----|-----|--------|
| < $95 | 0 | 200 | **+200** | 매수 (전량) |
| $95 ~ $97 | 100 | 200 | **+100** | 순매수 |
| $97 ~ $98 | 250 | 200 | **-50** | 순매도 |
| $98 ~ $100 | 450 | 200 | **-250** | 순매도 (퉁치기 결과) |
| > $100 | 450 | 0 | **-450** | 매도 (전량) |

Note: net flips from positive to negative between $95~$97 and $97~$98.

## Algorithm: Cumulative Sum Netting

The orderBook page uses this simplified algorithm for single-buy + multiple-sell netting:

```python
def netting_ranges(buy_price, buy_qty, sell_orders):
    """
    buy_price: float - buy LOC limit price
    buy_qty: int - buy quantity
    sell_orders: list of (price, qty) - nettable sells only (sell_price ≤ buy_price)

    Returns: list of price-range scenarios
    """
    # Group sells at same price, sort ascending
    from collections import defaultdict
    sell_groups = defaultdict(float)
    for price, qty in sell_orders:
        sell_groups[price] += qty
    sorted_sells = sorted(sell_groups.items())

    ranges = []
    cum_sell = 0

    # ① Below all sell prices: only buy executes
    min_sell = sorted_sells[0][0]
    ranges.append(("매수 (전량)", min_sell - 0.01, buy_qty,
                    f"종가 < ${min_sell} 시 매도미체결"))

    # ② Each sell price boundary: cumulative sell increases
    for i, (sp, sq) in enumerate(sorted_sells):
        cum_sell += sq
        net = buy_qty - cum_sell

        # Last range = netted result (already shown in main order sheet)
        if i == len(sorted_sells) - 1:
            continue

        next_sp = sorted_sells[i + 1][0]
        label = "순매수" if net > 0 else "순매도"
        ranges.append((label, sp, abs(net),
                        f"종가 ${sp}~${next_sp} 구간"))

    # ③ Above buy price: buy doesn't execute
    ranges.append(("매도 (전량)", buy_price + 0.01, cum_sell,
                    f"종가 > ${buy_price} 시 매수미체결"))

    return ranges
```

**Key insight**: `net = buy_qty - cumsum(sell_qtys)` — one subtraction per boundary.

## Application to Order Book Page

In the orderBook page, we display LOC orders for the next trading day:

**Buy Orders:**
- Base buy: `prev_close * (1 + buy_cond_pct%)`
- Spread buys: lower prices for "what if it drops more" scenarios

**Sell Orders:**
- TP (Take Profit): `entry_price * (1 + tp_pct%)`
- Expiration: At `prev_close` when max holding period exceeded
- SL (Stop Loss): `entry_price * (1 - sl_pct%)`

### Order Sheet Display

The order sheet shows three layers:

1. **Netted orders** — the base result after netting in the overlapping zone
2. **Price-range spread rows** — all scenarios outside the overlapping zone:
   - `매수 (전량)`: close < min sell price → only buy executes
   - `순매수` / `순매도`: intermediate ranges (multiple sells at different prices)
   - `매도 (전량)`: close > buy price → only sell executes
3. **Spread buy rows** — additional buy scenarios at lower prices (excluded from netting)

### Example: Single Sell

```
Orders:
- Buy at $65.11 for 28 shares
- Sell (TP) at $63.73 for 29 shares

Order Sheet after netting:
┌──────────────┬──────────┬──────┬──────────────────────────────────────┐
│ 구분         │ 주문가   │ 수량 │ 비고                                 │
├──────────────┼──────────┼──────┼──────────────────────────────────────┤
│ 매도 (전량)  │ $65.12   │ 29주 │ 종가 > $65.11 시 매수미체결 → 전량매도│
│ 매도 (TP)    │ $63.73   │  1주 │ 퉁치기 후 순매도 (종가 $63.73~$65.11)│
│ 매수 (전량)  │ $63.72   │ 28주 │ 종가 < $63.73 시 매도미체결 → 전량매수│
└──────────────┴──────────┴──────┴──────────────────────────────────────┘
```

### Example: Multiple Sells

```
Orders:
- Buy at $100 for 500 shares
- Sell (TP) at $95 for 100 shares
- Sell (TP) at $97 for 150 shares
- Sell (만료) at $98 for 200 shares

Order Sheet after netting:
┌──────────────┬──────────┬───────┬──────────────────────────────────────────┐
│ 구분         │ 주문가   │ 수량  │ 비고                                     │
├──────────────┼──────────┼───────┼──────────────────────────────────────────┤
│ 매도 (전량)  │ $100.01  │ 450주 │ 종가 > $100 시 매수미체결 → 전량매도      │
│ 매수         │ $100.00  │  50주 │ 퉁치기 후 순매수 (종가 $98~$100)         │
│ 순매수       │  $97.00  │ 250주 │ 종가 $97~$98 구간 (매도 250주 체결)       │
│ 순매수       │  $95.00  │ 400주 │ 종가 $95~$97 구간 (매도 100주 체결)       │
│ 매수 (전량)  │  $94.99  │ 500주 │ 종가 < $95 시 매도미체결 → 전량매수       │
└──────────────┴──────────┴───────┴──────────────────────────────────────────┘
```

### Netting Rules for Order Book

1. **Compare each sell order's price against the base buy price**
2. **Only net when sell_price ≤ buy_price**
3. **Spread buy orders** (e.g., "매수 (-3%)") are excluded from netting (scenario rows)
4. **SL orders** are displayed separately in a collapsible panel
5. **Non-nettable sells** (sell_price > buy_price) remain as independent orders

## Implementation Notes

1. **Netting is display-only**: actual cash/lots settlement is independent in the engine
2. **Spread buy rows** are excluded from netting (they're "what if" scenarios)
3. **SL orders** may have sell_price > buy_price (no netting) or sell_price < buy_price (netting possible)
4. **TP orders** typically have sell_price > entry_price, but may still be ≤ today's buy limit price
5. **Expiration sells** at prev_close will likely overlap with buy orders at `prev_close * (1 + X%)`
6. **매도 (전량) 주문가** uses `buy_price + $0.01` to represent "above buy price"; **매수 (전량) 주문가** uses `min_sell_price - $0.01` to represent "below all sells"
