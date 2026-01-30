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

### Case 4: Multiple Orders (Complex Case)

```
Buy at $100 for 500 shares
Buy at $95 for 300 shares
Sell at $98 for 200 shares
Sell at $102 for 400 shares
```

**Analysis by price range:**

| Close Price Range | Active Buy Orders | Active Sell Orders | Net Position |
|-------------------|-------------------|--------------------| ------------|
| < $95 | 500 + 300 = 800 | 0 | Net buy 800 |
| $95 - $98 | 500 | 0 | Net buy 500 |
| $98 - $100 | 500 | 200 | **Net buy 300** (netting!) |
| $100 - $102 | 0 | 200 | Net sell 200 |
| ≥ $102 | 0 | 200 + 400 = 600 | Net sell 600 |

## Algorithm for Price-Range Netting

```python
def calculate_netting(buy_orders, sell_orders):
    """
    buy_orders: list of (price, quantity) - buy if close <= price
    sell_orders: list of (price, quantity) - sell if close >= price

    Returns: list of net orders that can be submitted
    """
    # Step 1: Collect all price points
    prices = sorted(set(
        [o[0] for o in buy_orders] +
        [o[0] for o in sell_orders]
    ))

    # Step 2: Analyze each price range
    results = []
    for i, price in enumerate(prices):
        # At this exact price point:
        # - Buys execute if buy_price >= current_price
        # - Sells execute if sell_price <= current_price

        active_buy_qty = sum(qty for p, qty in buy_orders if p >= price)
        active_sell_qty = sum(qty for p, qty in sell_orders if p <= price)

        net_qty = active_buy_qty - active_sell_qty

        if i == 0:
            prev_net = active_buy_qty  # Below lowest price, only highest buys

        results.append({
            'price': price,
            'buy_qty': active_buy_qty,
            'sell_qty': active_sell_qty,
            'net_qty': net_qty,
            'netting_occurs': active_buy_qty > 0 and active_sell_qty > 0
        })

    return results
```

## Application to Order Book Page

In the orderBook page, we display LOC orders for the next trading day:

**Buy Orders:**
- Base buy: `prev_close * (1 + buy_cond_pct%)`
- Spread buys: lower prices for "what if it drops more" scenarios

**Sell Orders:**
- TP (Take Profit): `entry_price * (1 + tp_pct%)`
- Expiration: At `prev_close` when max holding period exceeded
- SL (Stop Loss): `entry_price * (1 - sl_pct%)`

### Netting Rules for Order Book

1. **Compare each sell order's price against each buy order's price**
2. **Only net when sell_price ≤ buy_price**
3. **Spread buy orders** (e.g., "매수 (-3%)") are at lower prices than base buy, so:
   - If base buy doesn't net with a sell, spread buys definitely won't
   - Spread buys represent alternative scenarios (if price drops more)
4. **Multiple sell orders** should be checked individually against buy orders

### Example

```
Current state:
- prev_close = $20.00
- Buy limit = $20.60 (buy_cond_pct = 3%)
- Open position: bought at $18.00, TP = $18.04

Orders:
- Buy at $20.60 for 10 shares (buy if close ≤ $20.60)
- Sell (TP) at $18.04 for 5 shares (sell if close ≥ $18.04)

Check: $18.04 (sell) ≤ $20.60 (buy)? YES → Netting possible!

If close is between $18.04 and $20.60:
- Both orders execute
- Net: buy 10 - sell 5 = net buy 5 shares
```

## Implementation Notes

1. **Always check price conditions** before applying netting
2. **Spread buy rows** should generally be excluded from netting (they're scenarios, not guaranteed orders)
3. **SL orders** may have sell_price > buy_price (no netting) or sell_price < buy_price (netting possible)
4. **TP orders** typically have sell_price > entry_price, but may still be ≤ today's buy limit price
5. **Expiration sells** at prev_close will likely overlap with buy orders at `prev_close * (1 + X%)`
