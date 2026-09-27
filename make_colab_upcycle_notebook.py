"""Build notebooks/04_colab_dense_to_moe_upcycling.ipynb from the local upcycling code.
Same fp16+GradScaler fallback as make_colab_notebook.py for GPUs without bf16 (T4)."""

import json
from pathlib import Path

ROOT = Path(__file__).parent

AMP_DEF = """
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
SCALER = torch.amp.GradScaler("cuda", enabled=(AMP_DTYPE == torch.float16))
"""

OLD_STEP = """        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
"""
NEW_STEP = """        if SCALER.is_enabled():
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


def patch(src, name, anchor):
    """Module-level AMP dtype/scaler after `anchor`; bf16 autocast -> AMP_DTYPE; scaled step."""
    assert anchor in src, name
    src = src.replace(anchor, anchor + AMP_DEF, 1)
    src = src.replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cuda", dtype=AMP_DTYPE)')
    assert OLD_STEP in src, name
    src = src.replace(OLD_STEP, NEW_STEP)
    compile(src, name, "exec")
    return src


def cell(code, kind="code"):
    return {"cell_type": kind, "metadata": {}, "source": code} | (
        {"outputs": [], "execution_count": None} if kind == "code" else {})


read = lambda f: (ROOT / f).read_text()
train_src = patch(read("train.py"), "train.py", "VOCAB = 8192\n")
upcycle_src = patch(read("upcycle.py"), "upcycle.py",
                    "from train import DATA, VOCAB, evaluate, get_batch\n")

fetch = '''
!pip -q install tokenizers huggingface_hub

import os, shutil
from huggingface_hub import hf_hub_download

os.makedirs("data", exist_ok=True)
dest = "data/fineweb-edu-00000.parquet"
if not os.path.exists("data/train.bin"):
    if not os.path.exists(dest):
        p = hf_hub_download("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet",
                            repo_type="dataset")
        try:                      # symlink to avoid a second 2.1 GB on disk
            os.symlink(p, dest)
        except OSError:
            shutil.copy(p, dest)
    print(f"{dest}: {os.path.getsize(dest)/1e9:.2f} GB")
'''

config = '''
# ---- experiment knobs -------------------------------------------------------------
TOTAL_TOKENS = 50e6   # one LR schedule over the whole budget
STOP_TOKENS  = 20e6   # dense phase ends and the MoE is created here
EXPERTS, TOPK = 8, 2
LR, BATCH = 7.5e-4, 16

# quick look instead (about 10 min on a T4): TOTAL_TOKENS, STOP_TOKENS = 10e6, 4e6
common = f"--total-tokens {TOTAL_TOKENS} --stop-tokens {STOP_TOKENS} --lr {LR} --batch {BATCH}"
print(common)
'''

equiv = '''
# Upcycling must be function-preserving: identical experts + renormalised top-k gates
# reproduce the dense MLP exactly, whatever the router says.
import torch
from model import GPTBaseline
from moe import upcycle

torch.manual_seed(0)
dense = GPTBaseline(8192, seq=512).cuda().eval()
moe = upcycle(dense, EXPERTS, TOPK).eval()
x = torch.randint(0, 8192, (2, 512), device="cuda")
with torch.no_grad():
    d, m = dense(x), moe(x)
print(f"max |dense - moe| = {(d - m).abs().max().item():.2e}")
n = lambda mod: sum(p.numel() for p in mod.parameters())
print(f"params: dense {n(dense)/1e6:.1f}M -> MoE {n(moe)/1e6:.1f}M")
del dense, moe
'''

summary = '''
import json
print(f"{'arm':20s} {'tokens':>12s} {'val@resume':>11s} {'final val (2M)':>15s} {'ppl':>7s} {'tok/s':>8s}")
for r in ["up_dense20M", "up_dense_cont", "up_moe8x2"]:
    m = json.load(open(f"runs/{r}/metrics.json"))
    vr = m.get("val_at_resume_this_model")
    print(f"{m['arm']:20s} {m['tokens_start']/1e6:5.1f}-{m['tokens_end']/1e6:4.1f}M "
          f"{(f'{vr:.4f}' if vr else '-'):>11s} {m['final_val_loss_2M']:15.4f} "
          f"{m['val_ppl']:7.1f} {m['tokens_per_s']/1e3:7.1f}k")
m = json.load(open("runs/up_moe8x2/metrics.json"))
print("\\nexpert divergence per layer (0 = still identical copies):",
      [round(v, 3) for v in m["expert_divergence_per_layer"]])
print("final expert load (share of routed slots, ideal", round(1 / m["experts"], 3), "):")
for i, row in enumerate(m["final_expert_load"]):
    print(f"  layer {i}: " + " ".join(f"{v:.3f}" for v in row))
'''

nb_cells = [
    cell("# Dense → MoE sparse upcycling (23M dense GPT → 8-expert top-2 MoE)\n\n"
         "1. Train a dense GPT (d=512, 6 layers, 23M params) on FineWeb-Edu for the first part of "
         "a single LR schedule and checkpoint it.\n"
         "2. **Upcycle**: every MLP is copied into E identical experts plus a small random router "
         "(top-k gates renormalised to sum 1), so the MoE starts with *exactly* the dense loss. "
         "AdamW moments are copied into each expert.\n"
         "3. Continue training the MoE for the rest of the schedule, next to a dense control run "
         "from the same checkpoint on identical batches.\n\n"
         "Runtime → Change runtime type → GPU. Uses bf16 on A100/L4, fp16 + GradScaler on T4.",
         "markdown"),
    cell("!nvidia-smi\nimport torch\n"
         "print('torch', torch.__version__, '| bf16 supported:', torch.cuda.is_bf16_supported())"),
    cell("%%writefile model.py\n" + read("model.py")),
    cell("%%writefile train.py\n" + train_src),
    cell("%%writefile moe.py\n" + read("moe.py")),
    cell("%%writefile upcycle.py\n" + upcycle_src),
    cell("%%writefile data_prep.py\n" + read("data_prep.py")),
    cell("%%writefile plot_upcycle.py\n" + read("plot_upcycle.py")),
    cell("## Data\nDownloads one FineWeb-Edu shard (~2.1 GB) and tokenizes 57M tokens with an "
         "8k BPE (a few minutes).", "markdown"),
    cell(fetch),
    cell("import os\nif not os.path.exists('data/train.bin'):\n    !python data_prep.py\n"
         "!ls -la data"),
    cell(config),
    cell("## Sanity check: the conversion does not change the model's outputs", "markdown"),
    cell(equiv),
    cell("## Phase 1: dense training (saves `runs/up_dense20M/ckpt.pt`)", "markdown"),
    cell("!python upcycle.py dense {common} --out runs/up_dense20M --log-every 200"),
    cell("## Phase 2a: upcycle to MoE and keep training\n"
         "The first log line compares the val loss of the dense checkpoint and the freshly "
         "upcycled MoE: they should match to ~1e-6.", "markdown"),
    cell("!python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode moe "
         "--experts {EXPERTS} --topk {TOPK} {common} --out runs/up_moe8x2 --log-every 200"),
    cell("## Phase 2b: dense control from the same checkpoint (same batches, same LR)",
         "markdown"),
    cell("!python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode dense "
         "{common} --out runs/up_dense_cont --log-every 200"),
    cell("## Results", "markdown"),
    cell(summary),
    cell("!python plot_upcycle.py\nfrom IPython.display import Image\n"
         "Image('curves_upcycle.png')"),
    cell("# optional: download results (checkpoint excluded)\n"
         "!zip -qr upcycle_results.zip runs/up_* curves_upcycle.png -x '*.pt'\n"
         "from google.colab import files\nfiles.download('upcycle_results.zip')"),
]

nb = {
    "nbformat": 4, "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "04_colab_dense_to_moe_upcycling.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": nb_cells,
}

out = ROOT / "notebooks" / "04_colab_dense_to_moe_upcycling.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({out.stat().st_size/1e3:.0f} KB, {len(nb_cells)} cells)")
