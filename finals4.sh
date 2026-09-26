#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
while ! grep -q "FINALS3 DONE" runs/finals3.log 2>/dev/null; do sleep 20; done
python train.py --arch rev_mid --batch 192 --lr 0.0013 --tokens 50e6 --out runs/r3_rev_mid_b192_lr0.0013 --log-every 50
echo "FINALS4 DONE"
