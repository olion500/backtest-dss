#!/bin/sh
# Refresh data/ in the background so boot is instant even when Yahoo is slow
# or blocked. Until the update lands, the app serves the build-time baseline
# CSVs plus the live gap-fill, which is already correct — just fetched live.
(
  python scripts/update_market_data.py \
    && echo "market data refreshed" \
    || echo "market data update failed; serving baseline + live gap-fill"
) &

exec uvicorn web.api.app.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
