"""Aggregate runs/*/metrics.json into a markdown summary + loss-curve plot."""
import json, math, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"

ORDER = ["r1b_baseline_b16_lr0.00075", "r2_rev_mid_b16", "r3_rev_mid_b192_lr0.0013",
         "r1_baseline_b16", "r3_rev_mid_b192_lr0.00065", "r3_rev_mid_b192_lr0.0026",
         "r3_rev_mid_b192_lr0.0052"]


def load(name):
    p = RUNS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def table(names, title):
    rows = []
    hdr = ("| run | arch | batch | steps | tokens | final train (EMA) | val loss | val ppl "
           "| tok/s | peak alloc MB | peak resv MB | wall min |")
    sep = "|" + "---|" * 12
    for n in names:
        m = load(n)
        if not m:
            continue
        rows.append(
            f"| {n} | {m['arch']} | {m['batch']} | {m['steps']} | {m['tokens']/1e6:.1f}M "
            f"| {m['final_train_loss_ema']:.4f} | {m['final_val_loss']:.4f} | {m['val_ppl']:.1f} "
            f"| {m['tokens_per_s']/1e3:.1f}k | {m['peak_mem_allocated_mb']:.0f} "
            f"| {m['peak_mem_reserved_mb']:.0f} | {m['wall_s']/60:.1f} |")
    return f"\n### {title}\n\n{hdr}\n{sep}\n" + "\n".join(rows) + "\n"


def clock_stats(log):
    p = RUNS / log
    if not p.exists():
        return None
    clk, tmp = [], []
    for line in p.read_text().splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) < 4 or "%" not in parts[3]:
            continue
        util = float(parts[3].replace("%", "").strip())
        if util < 50:            # only count while the GPU is actually training
            continue
        tmp.append(float(parts[0]))
        clk.append(float(parts[1].replace("MHz", "").strip()))
    if not clk:
        return None
    clk = np.array(clk)
    return {"log": log, "n": len(clk), "clk_max": clk.max(), "clk_first10_mean": clk[:10].mean(),
            "clk_last50_mean": clk[-50:].mean(), "clk_median": float(np.median(clk)),
            "temp_max": max(tmp)}


def main():
    out = ["# Results\n"]
    main_runs = [n for n in ORDER if (RUNS / n / "metrics.json").exists()]
    out.append(table(main_runs, "Main runs (50M tokens each)"))
    s3 = sorted(d.name for d in RUNS.iterdir() if d.name.startswith(("s3_", "s2_baseline")) and (d / "metrics.json").exists())
    if s3:
        out.append(table(s3, "Variant + LR selection, 10M tokens, batch 16 (AFTER the autocast-cache fix)"))
    buggy = sorted(d.name for d in RUNS.iterdir() if d.name.startswith("buggy_") and (d / "metrics.json").exists())
    if buggy:
        out.append(table(buggy, "Superseded: same sweeps BEFORE the fix (block weights were frozen)"))
    out.append("\n### GPU clock behaviour (throttling)\n")
    out.append("| log | samples>50% util | peak SM MHz | first-10 mean | last-50 mean | median | max temp C |")
    out.append("|" + "---|" * 7)
    for log in sorted(p.name for p in RUNS.glob("clocks_*.log")):
        s = clock_stats(log)
        if s:
            out.append(f"| {s['log']} | {s['n']} | {s['clk_max']:.0f} | {s['clk_first10_mean']:.0f} "
                       f"| {s['clk_last50_mean']:.0f} | {s['clk_median']:.0f} | {s['temp_max']:.0f} |")
    txt = "\n".join(out) + "\n"
    (ROOT / "RESULTS.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
