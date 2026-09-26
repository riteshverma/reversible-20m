"""Loss curves for the three main 50M-token runs, plotted against tokens consumed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"


def ema(v, a=0.02):
    out = np.empty_like(v)
    acc = v[0]
    for i, x in enumerate(v):
        acc = (1 - a) * acc + a * x
        out[i] = acc
    return out


def main(specs, fname, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, label, color in specs:
        d = RUNS / name
        if not (d / "curve.npy").exists():
            continue
        c = np.load(d / "curve.npy")
        m = json.loads((d / "metrics.json").read_text())
        toks = (c[:, 0] + 1) * m["batch"] * m["seq"] / 1e6
        ax.plot(toks, ema(c[:, 1]), label=f"{label} (val {m['final_val_loss']:.3f})",
                color=color, lw=1.8)
    ax.set_xlabel("tokens seen (M)")
    ax.set_ylabel("training loss (EMA)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_ylim(3.4, 7.0)
    fig.tight_layout()
    fig.savefig(ROOT / fname, dpi=140)
    print("wrote", fname)


if __name__ == "__main__":
    main([
        ("r1b_baseline_b16_lr0.00075", "baseline, b=16, lr 7.5e-4", "#2b6cb0"),
        ("r2_rev_mid_b16", "rev_mid (reversible), b=16, lr 7.5e-4", "#c05621"),
        ("r3_rev_mid_b192_lr0.0013", "rev_mid, b=192 (max), lr 1.3e-3", "#2f855a"),
        ("r1_baseline_b16", "baseline, b=16, lr 3e-3 (mistuned)", "#a0aec0"),
    ], "curves_main.png", "20M-param GPT, 50M tokens, RTX 3060 Laptop 6GB")
