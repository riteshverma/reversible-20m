#!/bin/sh
# Stronger load balancing: same dense 20M-token checkpoint, aux-loss coefficient 0.05 and 0.1
# (baseline run up_moe8x2 used 0.01).
set -e
for c in 0.05 0.1; do
  python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode moe --experts 8 --topk 2 \
    --aux-coef $c --out runs/up_moe8x2_aux$c
done
