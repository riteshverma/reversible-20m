#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
while ! grep -q "FINALS4 DONE" runs/finals4.log 2>/dev/null; do sleep 20; done
# baseline is still improving at the edge of its grid; rev_mid already has an interior
# minimum. Bound both so the best-vs-best comparison is honest.
python train.py --arch baseline  --batch 16 --lr 0.0004  --tokens 10e6 --out runs/s3_baseline_0.0004  --log-every 600
python train.py --arch baseline  --batch 16 --lr 0.0002  --tokens 10e6 --out runs/s3_baseline_0.0002  --log-every 600
python train.py --arch rev_euler --batch 16 --lr 0.0002  --tokens 10e6 --out runs/s3_rev_euler_0.0002 --log-every 600
echo "FINALS5 DONE"
