#!/bin/bash
# launch.sh <family> <target> <trials_per_worker> <workers> <train_start> <train_end> <tag>
cd "$(dirname "$0")"; mkdir -p studies
for i in $(seq 1 $4); do ../.venv/bin/python opt.py $1 $2 $3 $5 $6 $7 & done; wait
