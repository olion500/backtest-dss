#!/bin/bash
# launch_mo.sh <family> <trials_per_worker> <workers> <train_start> <train_end> <tag>
cd "$(dirname "$0")"; mkdir -p studies
for i in $(seq 1 $3); do ../.venv/bin/python opt_mo.py $1 $2 $4 $5 $6 & done; wait
