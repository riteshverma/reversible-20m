"""Steady-state thermal benchmark.

On this laptop the SM clock decays from ~1900 MHz to ~800 MHz as the GPU heats,
so cold-start tok/s is meaningless and even interleaved A/B inherits whatever
thermal state the machine happens to be in. This script:

  1. heat-soaks the GPU with a tensor-core burn until temperature plateaus,
  2. records which limiter is active while hot (HW thermal / SW power cap),
  3. only then runs the interleaved per-arch bursts, sampling SM clocks
     continuously, and reports both absolute and clock-normalized throughput.

tok/s per MHz is thermal-state invariant: it is the same number measured cold
or hot, which makes it the fair basis for architecture comparison here.

Clock lock (deterministic clocks, needs an ADMIN shell):
    nvidia-smi -lgc 850,850     # lock
    nvidia-smi -rgc             # restore
"""

import argparse
import json
import subprocess
import threading
import time

import torch

from model import ARCHS

Q = "temperature.gpu,clocks.sm,power.draw"


def smi(query=Q):
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    return [float(v) for v in out.split(",")]


class ClockLogger(threading.Thread):
    """Continuously sample temp/clock/power into a timestamped list."""

    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append((time.time(), *smi()))
            except Exception:
                pass
            time.sleep(self.interval)

    def clock_between(self, t0, t1):
        cs = [c for (t, _, c, _) in self.samples if t0 <= t <= t1]
        return sum(cs) / len(cs) if cs else float("nan")


def burn_until_plateau(max_s=900, min_s=240, tol=1.0, poll=10):
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    temps = []
    reasons = set()
    while time.time() - t0 < max_s:
        t_end = time.time() + poll
        while time.time() < t_end:
            c = a @ b
        torch.cuda.synchronize()
        temp, clk, pwr = smi()
        temps.append(temp)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks_event_reasons.active,"
                 "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
                 "clocks_event_reasons.sw_power_cap", "--format=csv,noheader"],
                capture_output=True, text=True).stdout.strip()
            reasons.add(r)
        except Exception:
            pass
        el = time.time() - t0
        print(f"soak {el:4.0f}s  {temp:.0f}C  {clk:.0f}MHz  {pwr:.0f}W", flush=True)
        if el >= min_s and len(temps) >= 3 and max(temps[-3:]) - min(temps[-3:]) <= tol:
            break
    del a, b, c
    torch.cuda.empty_cache()
    return temps[-1], reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="baseline,rev_mid,rev_euler")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--skip-soak", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev, V = "cuda", 8192
    archs = args.archs.split(",")

    print(f"start: {smi()}")
    if not args.skip_soak:
        steady_temp, reasons = burn_until_plateau()
        print(f"steady state: {steady_temp:.0f}C; limiter flags observed:\n  " +
              "\n  ".join(sorted(reasons)), flush=True)
    else:
        steady_temp, reasons = None, set()

    models, opts = {}, {}
    for a in archs:
        models[a] = ARCHS[a](V, seq=args.seq).to(dev)
        opts[a] = torch.optim.AdamW(models[a].parameters(), lr=1e-5, fused=True)
    x = torch.randint(0, V, (args.batch, args.seq), device=dev)
    y = torch.randint(0, V, (args.batch, args.seq), device=dev)

    def burst(a, n):
        m = models[a]
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y, ce_chunk=8192)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opts[a].step()
            opts[a].zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return n * args.batch * args.seq / (time.time() - t0)

    log = ClockLogger()
    log.start()
    for a in archs:
        burst(a, 6)                      # warm-up, not measured

    res = {a: [] for a in archs}         # (tok_s, mean_mhz)
    for r in range(args.rounds):
        for a in archs:
            t0 = time.time()
            tps = burst(a, args.steps)
            mhz = log.clock_between(t0, time.time())
            res[a].append((tps, mhz))
        print(f"round {r+1}: " + "  ".join(
            f"{a} {res[a][-1][0]/1e3:.1f}k@{res[a][-1][1]:.0f}MHz" for a in archs), flush=True)
    log.stop.set()

    summary = {"batch": args.batch, "steady_temp_c": steady_temp,
               "limiter_flags": sorted(reasons), "per_arch": {}}
    base = None
    for a in archs:
        tps = sum(t for t, _ in res[a]) / len(res[a])
        mhz = sum(c for _, c in res[a]) / len(res[a])
        if a == "baseline":
            base = tps
        summary["per_arch"][a] = {
            "tok_s": round(tps), "mean_sm_mhz": round(mhz),
            "tok_s_per_mhz": round(tps / mhz, 1),
            "ratio_vs_baseline": round(base / tps, 3) if base and a != "baseline" else 1.0,
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
