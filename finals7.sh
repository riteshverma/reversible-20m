#!/bin/bash
cd "c:/Users/Monisha Srivastava/.zcode/workspace/default/reversible-20m"
while ! grep -q "FINALS6 DONE" runs/finals6.log 2>/dev/null; do sleep 20; done
python speed_ab.py --batch 16 --rounds 5 --steps 12 > runs/speed_ab_b16.json 2>&1
python revcheck.py --arch rev_mid --steps 0   --bf16 1 > runs/revcheck_mid_init.json 2>&1
python revcheck.py --arch rev_mid --steps 300 --bf16 1 > runs/revcheck_mid_trained.json 2>&1
python revcheck.py --arch rev_euler --steps 300 --bf16 1 > runs/revcheck_euler_trained.json 2>&1
echo "FINALS7 DONE"
