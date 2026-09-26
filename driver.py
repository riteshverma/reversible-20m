"""Post-fix experiment chain: variant/LR sweep -> Run 2 (b=16) -> Run 3 (b=max)."""
import json, math, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable


def run(arch, batch, lr, tokens, out, log_every=400):
    cmd = [PY, "train.py", "--arch", arch, "--batch", str(batch), "--lr", str(lr),
           "--tokens", str(tokens), "--out", out, "--log-every", str(log_every)]
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads((ROOT / out / "metrics.json").read_text())


# ---- 1. variant + LR selection on a 10M-token horizon, batch 16 -------------------
sweep = {}
for arch, lrs in [("rev_euler", [0.0015, 0.003, 0.006]), ("rev_mid", [0.003, 0.006])]:
    for lr in lrs:
        m = run(arch, 16, lr, 10e6, f"runs/s3_{arch}_{lr}")
        sweep[(arch, lr)] = m["final_val_loss"]
        print(f"    -> {arch} lr={lr}: val {m['final_val_loss']:.4f}", flush=True)

print("\nSWEEP3:", {f"{a}@{l}": round(v, 4) for (a, l), v in sweep.items()}, flush=True)
best_arch, best_lr = min(sweep, key=sweep.get)
print(f"WINNER: {best_arch} lr={best_lr}", flush=True)

# ---- 2. Run 2: reversibility at the baseline's batch ------------------------------
run(best_arch, 16, best_lr, 50e6, "runs/r2_rev_b16", log_every=500)

# ---- 3. Run 3: reversibility at the maximum batch that fits 6GB -------------------
BMAX = 192
sqrt_lr = round(best_lr * math.sqrt(BMAX / 16), 5)
for lr in sorted({sqrt_lr, round(sqrt_lr / 2, 5)}):
    run(best_arch, BMAX, lr, 50e6, f"runs/r3_rev_b{BMAX}_lr{lr}", log_every=50)

print("DRIVER DONE", flush=True)
