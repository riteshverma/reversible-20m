"""Repeat the three final training configs under the thermal guard.

Per run: wait until the GPU returns to a cool baseline (<=65C, capped wait);
if it cannot get back under ~72C the conditions are the same as the documented
throttled regime, so the run is SKIPPED, not forced. Each started run trains
only while the GPU is in its cool/boost regime and is stopped by the guard the
moment throttling appears (train.py --thermal-guard).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent

RUNS = [
    ("baseline", 16,  7.5e-4, "g1_baseline_b16"),
    ("rev_mid",  16,  7.5e-4, "g2_rev_mid_b16"),
    ("rev_mid",  192, 1.3e-3, "g3_rev_mid_b192"),
]


def smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    return [float(v) for v in out.split(",")]


def wait_cool(target=65.0, cap_s=1200):
    t0 = time.time()
    while time.time() - t0 < cap_s:
        temp, _ = smi()
        if temp <= target:
            return True, temp, time.time() - t0
        time.sleep(15)
    return False, smi()[0], time.time() - t0


def main():
    results = []
    for arch, bs, lr, name in RUNS:
        ok, temp, waited = wait_cool()
        entry = {"run": name, "arch": arch, "batch": bs, "lr": lr,
                 "start_temp_c": temp, "cooldown_s": round(waited)}
        if not ok and temp > 72:
            entry.update(status="SKIPPED_SAME_HEAT",
                         note=f"GPU stuck at {temp:.0f}C after cooldown cap; "
                              "same heat conditions as throttled regime")
            results.append(entry)
            print(f"{name}: SKIPPED (GPU at {temp:.0f}C, cannot return to cool baseline)",
                  flush=True)
            continue

        outdir = ROOT / "runs" / name
        cmd = [sys.executable, "train.py", "--arch", arch, "--batch", str(bs),
               "--lr", str(lr), "--tokens", "50e6", "--out", str(outdir),
               "--log-every", "10", "--thermal-guard"]
        print(f"{name}: starting at {temp:.0f}C (cooled {waited:.0f}s)", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        (outdir / "stdout.log").write_text(proc.stdout[-8000:] + "\n--- stderr ---\n" + proc.stderr[-2000:])

        m = {}
        mp = outdir / "metrics.json"
        if mp.exists():
            m = json.loads(mp.read_text())
        entry.update(status="STOPPED_THERMAL" if m.get("stopped_reason") else "COMPLETED",
                     steps_done=m.get("steps_done"), tokens_done=m.get("tokens_done"),
                     tok_s_untill_stop=m.get("tokens_per_s"), wall_s=m.get("wall_s"),
                     stopped_reason=m.get("stopped_reason"),
                     final_train_loss_ema=m.get("final_train_loss_ema"))
        g = outdir / "guard.json"
        if g.exists():
            samples = json.loads(g.read_text()).get("samples", [])
            if samples:
                entry["max_temp_c"] = max(s["temp"] for s in samples)
                clks = [s["clk"] for s in samples if s["clk"] > 300]
                entry["min_clk_mhz"] = min(clks) if clks else None
        results.append(entry)
        print(f"{name}: {entry['status']} at {entry.get('tokens_done', 0)/1e6:.1f}M tokens "
              f"({entry.get('stopped_reason') or 'completed'})", flush=True)

    summary_path = ROOT / "runs" / "guarded_repeat_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
