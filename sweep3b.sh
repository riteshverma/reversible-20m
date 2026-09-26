#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
for spec in "rev_euler 0.00075" "rev_mid 0.00075" "rev_mid 0.0015" "baseline 0.0015" "rev_euler 0.0004"; do
  set -- $spec; a=$1; lr=$2
  python train.py --arch $a --batch 16 --lr $lr --tokens 10e6 --out runs/s3_${a}_${lr} --log-every 600
done
echo "SWEEP3B DONE"
