#!/bin/bash
# Train 2011-2022 (real SOXL, includes 2022), test 2023~ afterwards in report_mo.py.
cd "$(dirname "$0")"
for f in rsi sma_vix ddq sma_dd vix_dd rsi_sma single; do
  ./launch_mo.sh $f 300 12 2011-01-01 2022-12-31 mo_$f 2>/dev/null; echo done $f
done
