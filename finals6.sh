#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
while ! grep -q "FINALS5 DONE" runs/finals5.log 2>/dev/null; do sleep 20; done
python train.py --arch rev_mid --batch 192 --lr 0.00065 --tokens 50e6 --out runs/r3_rev_mid_b192_lr0.00065 --log-every 50
echo "FINALS6 DONE"
