#!/bin/bash
# Re-run the pair search for sma_vix with a fixed-shape search space (FLAT_SL=1).
cd "$(dirname "$0")"
for i in $(seq 1 12); do FLAT_SL=1 ../.venv/bin/python opt_mix.py sma_vix 250 2011-01-01 2022-12-31 mix2_sma_vix studies/seed_sma_vix0_flat.json 2>/dev/null & sleep 0.5; done; wait
echo done
