#!/bin/sh
# Dense 20M tokens -> fork: dense control vs 8-expert top-2 upcycled MoE, 30M tokens each.
set -e
python upcycle.py dense --stop-tokens 20e6 --out runs/up_dense20M
python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode moe --experts 8 --topk 2 --out runs/up_moe8x2
python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode dense --out runs/up_dense_cont
