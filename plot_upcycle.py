"""Plot the dense -> MoE upcycling runs: curves_upcycle.png."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).parent / "runs"
TOK = 16 * 512 / 1e6                           # M tokens per step
DENSE, MOE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def smooth(y, a=0.98):
    out, s = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        s = a * s + (1 - a) * v
        out[i] = s
    return out


def load(name):
    d = R / name
    return np.load(d / "curve.npy"), np.load(d / "evals.npy"), json.loads((d / "metrics.json").read_text())


c1, e1, m1 = load("up_dense20M")
cd, ed, md = load("up_dense_cont")
cm, em, mm = load("up_moe8x2")
conv = m1["tokens_end"] / 1e6
end = md["tokens_end"] / 1e6
MOE_LABEL = f"MoE {mm['experts']}x top-{mm['topk']}"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) training CE over the whole schedule
ax = axs[0]
dense_full = np.concatenate([c1, cd])
ax.plot(dense_full[:, 0] * TOK, smooth(dense_full[:, 1]), color=DENSE, lw=2, label="dense (23M)")
# MoE curve carries the dense EMA state across the conversion so the smoothing is continuous
moe_full = np.concatenate([c1, cm])
s = smooth(moe_full[:, 1])[len(c1):]
ax.plot(cm[:, 0] * TOK, s, color=MOE, lw=2, label=MOE_LABEL + " (upcycled)")
ax.axvline(conv, color=MUTED, lw=1, ls="--")
ax.text(conv, 6.3, " upcycled here", color=MUTED, va="top")
ax.set_ylim(3.4, 6.5)
ax.set_title("Training cross-entropy (EMA)", loc="left", color=INK)

# (b) validation loss after conversion
ax = axs[1]
ax.plot(e1[:, 0] * TOK, e1[:, 1], color=DENSE, lw=2, marker="o", ms=3)
ax.plot(ed[:, 0] * TOK, ed[:, 1], color=DENSE, lw=2, marker="o", ms=3, label="dense")
ax.plot(em[:, 0] * TOK, em[:, 1], color=MOE, lw=2, marker="o", ms=3, label=MOE_LABEL)
ax.axvline(conv, color=MUTED, lw=1, ls="--")
ax.set_xlim(conv - 0.3 * conv, end * 1.01)
lo = min(ed[:, 1].min(), em[:, 1].min())
ax.set_ylim(lo - 0.05, e1[e1[:, 0] * TOK >= 0.7 * conv, 1].max() + 0.05)
for e, col in ((ed, DENSE), (em, MOE)):
    ax.annotate(f"{e[-1, 1]:.3f}", (e[-1, 0] * TOK, e[-1, 1]), xytext=(4, 0),
                textcoords="offset points", color=INK, va="center")
ax.set_title("Validation loss (0.5M-token slice)", loc="left", color=INK)

# (c) MoE minus dense on identical eval points
ax = axs[2]
steps = np.intersect1d(ed[:, 0], em[:, 0])
gap = np.array([em[em[:, 0] == s, 1][0] - ed[ed[:, 0] == s, 1][0] for s in steps])
ax.axhline(0, color=MUTED, lw=1)
ax.plot(steps * TOK, gap, color=MOE, lw=2, marker="o", ms=3)
ax.annotate(f"{gap[-1]:+.3f}", (steps[-1] * TOK, gap[-1]), xytext=(4, 0),
            textcoords="offset points", color=INK, va="center")
ax.set_title("Val loss: MoE minus dense (below 0 = MoE better)", loc="left", color=INK)

for ax in axs:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("tokens seen (M)")
axs[0].set_ylabel("loss (nats/token)")
axs[0].legend(frameon=False)
axs[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "curves_upcycle.png", dpi=130)
print("wrote curves_upcycle.png")
