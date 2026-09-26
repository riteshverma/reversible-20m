#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
T=10e6
for spec in "rev_euler 0.0015" "rev_euler 0.003" "rev_euler 0.006" "rev_mid 0.0015" "rev_mid 0.003" "rev_mid 0.006" "baseline 0.003"; do
  set -- $spec
  a=$1; lr=$2
  echo "=== $a lr=$lr ==="
  python train.py --arch $a --batch 16 --lr $lr --tokens $T --out runs/s2_${a}_${lr} --log-every 400
done
echo "SWEEP2 DONE"
