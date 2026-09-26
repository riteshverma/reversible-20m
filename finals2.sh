#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
while ! grep -q "SWEEP3B DONE" runs/sweep3b.log; do sleep 20; done
echo "=== sweep3b finished, extending grid ==="
python train.py --arch rev_mid --batch 16 --lr 0.0004 --tokens 10e6 --out runs/s3_rev_mid_0.0004 --log-every 600
echo "=== RUN 2: reversibility at the baseline batch ==="
python train.py --arch rev_mid --batch 16 --lr 0.00075 --tokens 50e6 --out runs/r2_rev_mid_b16 --log-every 500
echo "=== RUN 3: reversibility at max batch (192) ==="
python train.py --arch rev_mid --batch 192 --lr 0.0026 --tokens 50e6 --out runs/r3_rev_mid_b192_lr0.0026 --log-every 50
python train.py --arch rev_mid --batch 192 --lr 0.0052 --tokens 50e6 --out runs/r3_rev_mid_b192_lr0.0052 --log-every 50
echo "FINALS2 DONE"
