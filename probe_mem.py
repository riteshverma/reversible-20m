"""Quick GPU memory probe: run 3 training steps at a given arch/batch, report peak memory."""

import argparse
import torch
from model import ARCHS

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=list(ARCHS))
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--seq", type=int, default=512)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
V = 8192
m = ARCHS[args.arch](V, seq=args.seq).to(dev)
x = torch.randint(0, V, (args.batch, args.seq), device=dev)
y = torch.randint(0, V, (args.batch, args.seq), device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
torch.cuda.reset_peak_memory_stats()
ok = True
for i in range(3):
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y, ce_chunk=8192)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    except torch.OutOfMemoryError:
        ok = False
        print(f"OOM at step {i}")
        break
if ok:
    alloc = torch.cuda.max_memory_allocated() / 2**20
    res = torch.cuda.max_memory_reserved() / 2**20
    print(f"OK arch={args.arch} batch={args.batch} peak_alloc={alloc:.0f}MB peak_reserved={res:.0f}MB")
