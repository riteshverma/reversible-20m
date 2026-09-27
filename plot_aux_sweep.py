"""Load-balancing-loss sweep for the upcycled MoE: curves_aux_sweep.png + a printed table."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "runs"
TOK = 16 * 512 / 1e6
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("0.01", "up_moe8x2", "#eb6834"),
        ("0.05", "up_moe8x2_aux0.05", "#1baf7a"),
        ("0.1", "up_moe8x2_aux0.1", "#4a3aa7")]


def load(name):
    d = R / name
    return np.load(d / "evals.npy"), json.loads((d / "metrics.json").read_text())


ed, md = load("up_dense_cont")
arms = [(c, *load(n), col) for c, n, col in ARMS if (R / n / "metrics.json").exists()]

print(f"{'aux coef':>8s} {'final val':>10s} {'vs dense':>9s} {'layer-0 max load':>17s} "
      f"{'worst layer max':>16s} {'min load':>9s}")
print(f"{'dense':>8s} {md['final_val_loss_2M']:10.4f}")
for c, e, m, _ in arms:
    ld = np.array(m["final_expert_load"])
    print(f"{c:>8s} {m['final_val_loss_2M']:10.4f} {m['final_val_loss_2M'] - md['final_val_loss_2M']:+9.4f} "
          f"{ld[0].max():17.3f} {ld.max(1).max():16.3f} {ld.min():9.3f}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axs[0]
ax.axhline(0, color=MUTED, lw=1)
ax.text(ed[-1, 0] * TOK, 0.002, "dense control ", color=MUTED, ha="right", va="bottom")
for c, e, m, col in arms:
    steps = np.intersect1d(ed[:, 0], e[:, 0])
    gap = np.array([e[e[:, 0] == s, 1][0] - ed[ed[:, 0] == s, 1][0] for s in steps])
    ax.plot(steps * TOK, gap, color=col, lw=2, marker="o", ms=3,
            label=f"aux coef {c}  (final {gap[-1]:+.3f})")
ax.set_title("Val loss: MoE minus dense (below 0 = MoE better)", loc="left", color=INK)
ax.set_xlabel("tokens seen (M)")
ax.set_ylabel("loss difference (nats/token)")
ax.legend(frameon=False)

ax = axs[1]
E = arms[0][2]["experts"]
ax.axhline(1 / E, color=MUTED, lw=1, ls="--")
ax.text(5.1, 1 / E, " even split", color=MUTED, va="bottom", ha="right")
for c, e, m, col in arms:
    ld = np.array(m["final_expert_load"])
    ax.plot(range(len(ld)), ld.max(1), color=col, lw=2, marker="o", ms=8, label=f"aux coef {c}")
ax.set_xticks(range(6))
ax.set_ylim(0, None)
ax.set_title("Busiest expert's share of routed slots (end)", loc="left", color=INK)
ax.set_xlabel("layer")
ax.set_ylabel("share of routed slots")

for ax in axs:
    ax.grid(True, color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "curves_aux_sweep.png", dpi=130)
print("wrote curves_aux_sweep.png")
