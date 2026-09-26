"""Trainer for the 20M-class reversible-vs-baseline experiments.

Examples:
  python train.py --arch baseline --batch 16 --tokens 50e6 --out runs/baseline_b16
  python train.py --arch rev_mid  --batch 16 --tokens 50e6 --out runs/revmid_b16
  python train.py --arch rev_mid  --batch 96 --tokens 50e6 --out runs/revmid_bmax --lr 4.2e-3
  python train.py --arch rev_mid  --probe-batch 128          # 3-step memory probe
"""

import argparse
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import ARCHS, count_params

ROOT = Path(__file__).parent
DATA = ROOT / "data"
VOCAB = 8192


def get_batch(data, idxs, seq, device):
    xs = np.stack([data[i : i + seq] for i in idxs])
    ys = np.stack([data[i + 1 : i + 1 + seq] for i in idxs])
    return (torch.from_numpy(xs.astype(np.int64)).to(device, non_blocking=True),
            torch.from_numpy(ys.astype(np.int64)).to(device, non_blocking=True))


@torch.no_grad()
def evaluate(model, val_data, seq, device, max_tokens=2_000_000, batch=8):
    model.eval()
    n = len(val_data) - seq - 1
    rng = np.random.default_rng(7)
    total, count = 0.0, 0
    done = 0
    while done < max_tokens:
        b = min(batch, (max_tokens - done) // seq)
        if b <= 0:
            break
        idxs = rng.integers(0, n, size=b)
        x, y = get_batch(val_data, idxs, seq, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        total += loss.item() * b * seq
        count += b * seq
        done += b * seq
    model.train()
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCHS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tokens", type=float, default=50e6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ce-chunk", type=int, default=8192)
    ap.add_argument("--probe-batch", type=int, default=0, help="run 3 steps at this batch, report memory, exit")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--thermal-guard", action="store_true",
                    help="stop the run when the GPU enters the throttled regime "
                         "(SW thermal slowdown flag active, or >=87C)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    out = Path(args.out) if args.out else ROOT / "runs" / f"{args.arch}_b{args.batch}"
    out.mkdir(parents=True, exist_ok=True)

    train_mm = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val_mm = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")

    model = ARCHS[args.arch](VOCAB, seq=args.seq).to(device)
    pcount = count_params(model)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

    steps = int(args.tokens // (args.batch * args.seq))
    warmup = max(30, int(steps * args.warmup_frac))

    if args.probe_batch:
        args.batch = args.probe_batch
        steps = 3

    print(f"arch={args.arch} params={pcount} batch={args.batch} seq={args.seq} steps={steps} lr={args.lr}", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    n_train = len(train_mm) - args.seq - 1

    guard_stop = threading.Event()
    guard_done = threading.Event()
    guard_state = {"reason": None}
    guard_samples = []

    def _guard_smi():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,"
             "clocks_event_reasons.sw_thermal_slowdown", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip()
        p = [v.strip() for v in out.split(",")]
        return float(p[0]), float(p[1]), float(p[2]), p[3] == "Active"

    def thermal_watchdog():
        hot, t0 = 0, time.time()
        while not guard_done.is_set():
            try:
                temp, clk, pwr, sw = _guard_smi()
            except Exception:
                time.sleep(2.0)
                continue
            guard_samples.append({"t": round(time.time() - t0, 1), "temp": temp,
                                  "clk": clk, "pwr": pwr, "sw_thermal": sw})
            # the documented throttled regime: SW thermal governor engaged (its flag,
            # or >=87C) at ANY clock level; require 2 consecutive hits
            throttled = temp >= 87 or sw
            hot = hot + 1 if throttled else 0
            if hot >= 2:
                guard_state["reason"] = (f"throttled: {temp:.0f}C {clk:.0f}MHz "
                                         f"sw_thermal_slowdown={sw}")
                guard_stop.set()
                return
            time.sleep(2.0)

    if args.thermal_guard:
        threading.Thread(target=thermal_watchdog, daemon=True).start()

    curve, mem_peak_alloc, mem_peak_res = [], 0, 0
    t_start = time.time()
    t_steady_start, tokens_steady = None, 0
    ema = None
    steps_done = steps

    for step in range(steps):
        lr = args.lr * min((step + 1) / warmup, 1.0) if step < warmup else \
             args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
        for group in opt.param_groups:
            group["lr"] = lr

        idxs = torch.randint(0, n_train, (args.batch,), generator=g).numpy()
        x, y = get_batch(train_mm, idxs, args.seq, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y, ce_chunk=args.ce_chunk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        lv = loss.item()
        ema = lv if ema is None else 0.95 * ema + 0.05 * lv
        curve.append((step, lv, lr))

        if step == 5:  # start steady-state timing after warm-up of allocator
            torch.cuda.synchronize()
            t_steady_start = time.time()
            tokens_steady = 0
        if step >= 5:
            tokens_steady += args.batch * args.seq

        mem_peak_alloc = max(mem_peak_alloc, torch.cuda.max_memory_allocated())
        mem_peak_res = max(mem_peak_res, torch.cuda.max_memory_reserved())

        if (step + 1) % args.log_every == 0 or step == steps - 1:
            el = time.time() - t_start
            tps = tokens_steady / max(time.time() - t_steady_start, 1e-9) if step >= 5 else 0
            print(f"step {step+1}/{steps} loss {lv:.4f} ema {ema:.4f} lr {lr:.2e} "
                  f"tok/s {tps/1e3:.1f}k elapsed {el:.0f}s", flush=True)

        if args.thermal_guard and guard_stop.is_set():
            steps_done = step + 1
            print(f"THERMAL GUARD: stopping at step {step+1}/{steps}: {guard_state['reason']}",
                  flush=True)
            break

    guard_done.set()
    wall = time.time() - t_start
    tps_steady = tokens_steady / max(time.time() - t_steady_start, 1e-9) if t_steady_start else 0.0
    stopped = args.thermal_guard and guard_state["reason"] is not None

    if stopped:
        val_loss, val_ppl = None, None
    else:
        val_loss = evaluate(model, val_mm, args.seq, device)
        val_ppl = math.exp(val_loss)
    metrics = {
        "arch": args.arch, "batch": args.batch, "seq": args.seq,
        "tokens_planned": steps * args.batch * args.seq, "steps_planned": steps,
        "steps_done": steps_done, "tokens_done": steps_done * args.batch * args.seq,
        "tokens": steps_done * args.batch * args.seq, "steps": steps_done,
        "lr": args.lr, "params": pcount,
        "final_train_loss_ema": ema,
        "final_val_loss": val_loss, "val_ppl": val_ppl,
        "tokens_per_s": tps_steady,
        "peak_mem_allocated_mb": mem_peak_alloc / 2**20,
        "peak_mem_reserved_mb": mem_peak_res / 2**20,
        "wall_s": wall, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "seed": args.seed,
        "stopped_reason": guard_state["reason"],
        "guard_samples_n": len(guard_samples),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / "curve.npy", np.array(curve))
    if args.thermal_guard:
        (out / "guard.json").write_text(json.dumps(
            {"reason": guard_state["reason"], "samples": guard_samples}, indent=1))
    if args.save_ckpt:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": steps}, out / "ckpt.pt")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
