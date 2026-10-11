#!/bin/bash
# Walk-forward (train 2002-2020, test 2021~) for the second batch of signal families.
cd "$(dirname "$0")"
for job in sma_dd:SOXL_X vix_dd:SOXL_X ddq:USD_X sma_vix:USD_X; do
  f=${job%%:*}; t=${job##*:}
  ./launch.sh $f $t 170 12 2002-01-01 2020-12-31 wf3_${t}_$f 2>/dev/null; echo done $t $f
done
