#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
# Run 2: reversibility at the SAME batch as the baseline run
python train.py --arch rev_euler --batch 16 --lr 6e-3 --tokens 50e6 --out runs/r2_rev_euler_b16 --log-every 500
# Run 3: reversibility pushed to the maximum batch that fits 6GB (192), two LRs
python train.py --arch rev_euler --batch 192 --lr 1.2e-2 --tokens 50e6 --out runs/r3_rev_euler_b192_lr1.2e-2 --log-every 50
python train.py --arch rev_euler --batch 192 --lr 2.4e-2 --tokens 50e6 --out runs/r3_rev_euler_b192_lr2.4e-2 --log-every 50
echo "FINAL RUNS DONE"
