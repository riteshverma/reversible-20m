"""The 10M sweep showed baseline prefers 1.5e-3 (4.952) over the 3e-3 that Run 1 used
(5.181). Probe one LR lower, then re-run the 50M baseline at its own best LR so the
headline comparison is best-vs-best rather than best-vs-mistuned."""
import glob, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
PY = sys.executable

while "FINALS2 DONE" not in (ROOT / "runs/finals2.log").read_text(errors="ignore"):
    time.sleep(20)

subprocess.run([PY, "train.py", "--arch", "baseline", "--batch", "16", "--lr", "0.00075",
                "--tokens", "10e6", "--out", "runs/s3_baseline_0.00075", "--log-every", "600"],
               cwd=ROOT, check=True)

cands = {}
for d in glob.glob(str(ROOT / "runs/s3_baseline_*")) + [str(ROOT / "runs/s2_baseline_0.003")]:
    p = os.path.join(d, "metrics.json")
    if os.path.exists(p):
        m = json.loads(open(p).read())
        cands[m["lr"]] = m["final_val_loss"]
print("baseline 10M sweep:", {k: round(v, 4) for k, v in sorted(cands.items())}, flush=True)
best = min(cands, key=cands.get)
print("baseline best lr:", best, flush=True)

subprocess.run([PY, "train.py", "--arch", "baseline", "--batch", "16", "--lr", str(best),
                "--tokens", "50e6", "--out", f"runs/r1b_baseline_b16_lr{best}", "--log-every", "500"],
               cwd=ROOT, check=True)
print("FINALS3 DONE", flush=True)
