#!/bin/bash
# Pair-with-current search: train 2011-2022, test 2023~ in report_mix.py.
cd "$(dirname "$0")"
run() {  # family seed
  for i in $(seq 1 12); do ../.venv/bin/python opt_mix.py $1 200 2011-01-01 2022-12-31 mix_$1 $2 & sleep 0.5; done; wait
}
run sma_vix studies/seed_sma_vix0.json; echo done sma_vix
for f in rsi_sma vix_dd ddq; do run $f; echo done $f; done
