"""Throughput-vs-batch sweep. Detects the point where the Windows driver starts
spilling VRAM into shared system memory (peak_reserved > physical), which shows up
as a throughput collapse rather than an OOM."""
import argparse, json, time
import torch
from model import ARCHS

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=list(ARCHS))
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=6)
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev, V = "cuda", 8192
free, total = torch.cuda.mem_get_info()

m = ARCHS[args.arch](V, seq=args.seq).to(dev)
x = torch.randint(0, V, (args.batch, args.seq), device=dev)
y = torch.randint(0, V, (args.batch, args.seq), device=dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)

try:
    for i in range(args.steps):
        if i == args.warmup:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, y, ce_chunk=8192)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = (args.steps - args.warmup) * args.batch * args.seq
    print(json.dumps({"arch": args.arch, "batch": args.batch, "ok": True,
                      "tok_s": n / dt, "step_ms": dt / (args.steps - args.warmup) * 1e3,
                      "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
                      "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
                      "phys_total_mb": total / 2**20, "phys_free_at_start_mb": free / 2**20}))
except torch.OutOfMemoryError:
    print(json.dumps({"arch": args.arch, "batch": args.batch, "ok": False, "err": "OOM"}))
