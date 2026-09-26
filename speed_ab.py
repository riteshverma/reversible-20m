"""Thermally-matched throughput A/B.

Comparing tok/s between separate long runs is unfair on this laptop: the GPU decays from
~1900MHz to ~800MHz as it heats, so whichever config ran on a cooler card looks faster.
Here both models live in one process and are interleaved in short bursts, so each round
sees near-identical thermal state and the ratio is meaningful even as absolute speed decays.
"""
import argparse, json, time

import torch

from model import ARCHS

ap = argparse.ArgumentParser()
ap.add_argument("--archs", default="baseline,rev_mid,rev_euler")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--rounds", type=int, default=5)
ap.add_argument("--steps", type=int, default=12)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev, V = "cuda", 8192
archs = args.archs.split(",")

models, opts = {}, {}
for a in archs:
    models[a] = ARCHS[a](V, seq=args.seq).to(dev)
    opts[a] = torch.optim.AdamW(models[a].parameters(), lr=1e-5, fused=True)
x = torch.randint(0, V, (args.batch, args.seq), device=dev)
y = torch.randint(0, V, (args.batch, args.seq), device=dev)


def burst(a, n):
    m, opt = models[a], opts[a]
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y, ce_chunk=8192)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return n * args.batch * args.seq / (time.time() - t0)


for a in archs:               # allocator / cudnn warm-up, not measured
    burst(a, 4)

res = {a: [] for a in archs}
for r in range(args.rounds):
    for a in archs:
        res[a].append(burst(a, args.steps))
    print(f"round {r+1}: " + "  ".join(f"{a} {res[a][-1]/1e3:.1f}k" for a in archs), flush=True)

print(json.dumps({
    "batch": args.batch,
    "per_arch_tok_s_mean": {a: sum(v) / len(v) for a, v in res.items()},
    "per_arch_tok_s_best": {a: max(v) for a, v in res.items()},
    "rounds": {a: [round(t) for t in v] for a, v in res.items()},
    "ratio_vs_baseline_mean": {a: (sum(res["baseline"]) / len(res["baseline"])) / (sum(v) / len(v))
                               for a, v in res.items()} if "baseline" in res else None,
}, indent=2))
