"""Write human-readable training logs next to each upcycling run's binary curves.

  runs/<run>/train_log.csv  step, tokens, train_ce, aux_loss, lr   (every optimizer step)
  runs/<run>/val_log.csv    step, tokens, val_loss                   (every 250 steps, 0.5M-token slice)
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
TOK_PER_STEP = 16 * 512

for d in sorted(p.parent for p in ROOT.glob("runs/**/evals.npy")):
    curve = np.load(d / "curve.npy")                      # (step, ce, aux, lr)
    with open(d / "train_log.csv", "w", newline="\n") as f:
        f.write("step,tokens,train_ce,aux_loss,lr\n")
        for step, ce, aux, lr in curve:
            f.write(f"{int(step) + 1},{(int(step) + 1) * TOK_PER_STEP},{ce:.5f},{aux:.5f},{lr:.4e}\n")
    evals = np.load(d / "evals.npy")                      # (step, val)
    with open(d / "val_log.csv", "w", newline="\n") as f:
        f.write("step,tokens,val_loss\n")
        for step, v in evals:
            f.write(f"{int(step)},{int(step) * TOK_PER_STEP},{v:.5f}\n")
    print(f"{d.relative_to(ROOT)}: {len(curve)} train rows, {len(evals)} val rows")
