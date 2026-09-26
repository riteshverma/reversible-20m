"""Build a self-contained Colab notebook from the (fixed) local experiment code.
Adds an fp16+GradScaler path for T4 GPUs (bf16 unsupported on Turing)."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent


def patched_train():
    src = (ROOT / "train.py").read_text()

    # parametrize autocast dtype
    src = src.replace(
        'with torch.autocast("cuda", dtype=torch.bfloat16):',
        'with torch.autocast("cuda", dtype=AMP_DTYPE):')

    # insert dtype + scaler right after device selection
    anchor = '    device = "cuda"\n'
    assert anchor in src
    # evaluate() is module-level and also autocasts, so these must be globals, not
    # locals of main() -- otherwise the eval at the END of a 25-minute run NameErrors.
    src = src.replace(anchor, anchor + """
    global AMP_DTYPE, SCALER
    AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    SCALER = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
    print(f"autocast dtype: {AMP_DTYPE}", flush=True)
""")

    old_step = """        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
"""
    new_step = """        if SCALER.is_enabled():
            SCALER.scale(loss).backward()
            SCALER.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            SCALER.step(opt)
            SCALER.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        opt.zero_grad(set_to_none=True)
"""
    assert old_step in src
    src = src.replace(old_step, new_step)
    compile(src, "train_colab.py", "exec")
    return src


def cell(code, kind="code"):
    return {"cell_type": kind, "metadata": {}, "source": code} | (
        {"outputs": [], "execution_count": None} if kind == "code" else {})


model_src = (ROOT / "model.py").read_text()
train_src = patched_train()
dataprep_src = (ROOT / "data_prep.py").read_text()
gradcheck_src = (ROOT / "gradcheck.py").read_text()

probes = '''
# Find the largest batch that fits THIS runtime (T4/L4/A100 differ a lot).
# --probe-batch ends by dumping metrics JSON, so parse that rather than the last line.
import subprocess, sys, json, re, torch

free, total = torch.cuda.mem_get_info()
print(f"free {free/2**20:.0f} MB / total {total/2**20:.0f} MB\\n")

def probe(arch, b):
    r = subprocess.run([sys.executable, "train.py", "--arch", arch,
                        "--probe-batch", str(b)], capture_output=True, text=True)
    m = re.search(r"\\{[\\s\\S]*\\}\\s*$", r.stdout)
    if r.returncode != 0 or not m:
        err = (r.stderr.strip().splitlines() or ["failed"])[-1]
        print(f"{arch:9s} b={b:4d}: FAILED/OOM  {err[:80]}")
        return False
    d = json.loads(m.group(0))
    print(f"{arch:9s} b={b:4d}: alloc {d['peak_mem_allocated_mb']:7.0f} MB   "
          f"reserved {d['peak_mem_reserved_mb']:7.0f} MB   {d['tokens_per_s']/1e3:6.1f}k tok/s")
    # a batch only really fits if reserved stays inside free VRAM
    return d["peak_mem_reserved_mb"] < free / 2**20

for arch, batches in [("baseline", [64, 128, 256]), ("rev_mid", [256, 384, 512, 640])]:
    print(f"=== {arch} ===")
    for b in batches:
        if not probe(arch, b):
            break
'''

fetch = '''
!pip -q install tokenizers huggingface_hub

import os, shutil
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)
dest = "data/fineweb-edu-00000.parquet"
if not os.path.exists(dest):
    p = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet",
                        repo_type="dataset")
    try:                      # symlink to avoid a second 2.1 GB on disk
        os.symlink(p, dest)
    except OSError:
        shutil.copy(p, dest)
print(f"{dest}: {os.path.getsize(dest)/1e9:.2f} GB")
'''

summary = """
import json, glob
print(f"{'run':28s} {'arch':9s} {'batch':>5s} {'val_loss':>9s} {'ppl':>8s} {'tok/s':>8s} {'peakMB':>7s}")
for f in sorted(glob.glob("runs/r*/metrics.json")):
    m = json.load(open(f))
    vl = f"{m['final_val_loss']:.4f}" if m.get("final_val_loss") else "n/a"
    pl = f"{m['val_ppl']:.1f}" if m.get("val_ppl") else "n/a"
    print(f"{f.split('/')[1]:28s} {m['arch']:9s} {m['batch']:5d} {vl:>9s} {pl:>8s} "
          f"{m['tokens_per_s']/1e3:7.1f}k {m['peak_mem_allocated_mb']:7.0f}")
"""

nb_cells = [
    cell("# 20M reversible LLM — 50M tokens on FineWeb-Edu (baseline vs reversible)\n"
         "\nSame code/configs as the local experiment: baseline / rev_mid (leapfrog midpoint) /\n"
         "rev_mid at max batch. fp16 autocast + GradScaler if bf16 is unavailable (T4).", "markdown"),
    cell("!nvidia-smi\nimport torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n"
         "print('torch', torch.__version__, '| bf16 supported:', torch.cuda.is_bf16_supported())"),
    cell("%%writefile model.py\n" + model_src),
    cell("%%writefile train.py\n" + train_src),
    cell("%%writefile data_prep.py\n" + dataprep_src),
    cell("## Fetch the corpus\n"
         "`data_prep.py` reads a local parquet shard, which does not exist on a fresh Colab "
         "runtime. This pulls the same FineWeb-Edu shard used locally (~2.1 GB, a few minutes) "
         "and puts it where `data_prep.py` expects it.", "markdown"),
    cell(fetch),
    cell("!python data_prep.py"),
    cell("%%writefile gradcheck.py\n" + gradcheck_src),
    cell("!python gradcheck.py"),
    cell(probes),
    cell("!python train.py --arch baseline --batch 16 --lr 7.5e-4 --tokens 50e6 "
         "--out runs/r1_baseline_b16 --log-every 200"),
    cell("!python train.py --arch rev_mid --batch 16 --lr 7.5e-4 --tokens 50e6 "
         "--out runs/r2_rev_mid_b16 --log-every 200"),
    cell("!python train.py --arch rev_mid --batch 192 --lr 1.3e-3 --tokens 50e6 "
         "--out runs/r3_rev_mid_b192 --log-every 100"),
    cell("# optional: push to the largest batch that fits this runtime "
         "(edit batch to the probe result)\n"
         "!python train.py --arch rev_mid --batch 512 --lr 2.1e-3 --tokens 50e6 "
         "--out runs/r4_rev_mid_bmax --log-every 50"),
    cell(summary),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "01_colab_train_50M.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    },
    "cells": nb_cells,
}

out = ROOT / "notebooks" / "01_colab_train_50M.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size/1e3:.0f} KB, {len(nb_cells)} cells)")
