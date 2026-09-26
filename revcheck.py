"""
Does the reversible backward stay exact under the conditions we actually TRAIN in
(GPU + bf16 autocast + trained weights), as opposed to the fp32/CPU/at-init setting
gradcheck.py covers?

Two measurements, taken at init and again after K optimizer steps:
  1. reconstruction error: forward once keeping every (a_i, b_i); then invert from the
     final pair exactly as RevEulerFn.backward does, and compare.
  2. gradient error vs a store-everything autograd reference run under the same autocast.
"""
import argparse, json

import numpy as np
import torch

from model import ARCHS, lm_loss

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="rev_euler", choices=list(__import__("model").ARCHS))
ap.add_argument("--steps", type=int, default=0, help="train this many steps before measuring")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--seq", type=int, default=512)
ap.add_argument("--lr", type=float, default=6e-3)
ap.add_argument("--bf16", type=int, default=1)
args = ap.parse_args()

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
dev, V = "cuda", 8192
CLS = ARCHS[args.arch]
m = CLS(V, seq=args.seq).to(dev)
autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(args.bf16))

data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
g = torch.Generator().manual_seed(0)


def batch():
    i = torch.randint(0, len(data) - args.seq - 1, (args.batch,), generator=g).numpy()
    x = np.stack([data[j: j + args.seq] for j in i]).astype(np.int64)
    y = np.stack([data[j + 1: j + 1 + args.seq] for j in i]).astype(np.int64)
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


if args.steps:
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), fused=True)
    for _ in range(args.steps):
        x, y = batch()
        with autocast():
            loss = m(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)

x, y = batch()
cos, sin = m.cos.to(dev)[: args.seq], m.sin.to(dev)[: args.seq]


def rel(a, b):
    """Relative error against the reference scale. Returns None when the reference is the
    all-zero seed tensor (leapfrog's x_{-1}), where a ratio is meaningless."""
    scale = b.abs().max()
    if scale.item() == 0.0:
        return None
    return ((a - b).abs().max() / scale).item()


def wmax(*vals):
    v = [x for x in vals if x is not None]
    return max(v) if v else None


# ---- 1. reconstruction error ------------------------------------------------------
recon = []
with torch.no_grad(), autocast():
    e = m.emb(x)
    if args.arch.startswith("rev_euler"):
        a, b = e, e
        keep = [(a, b)]
        for blk in m.blocks:
            a = a + blk.f(b, cos, sin)
            b = b + blk.g(a, cos, sin)
            keep.append((a, b))
        a_r, b_r = keep[-1]
        for i, blk in list(enumerate(m.blocks))[::-1]:
            b_prev = b_r - blk.g(a_r, cos, sin)
            a_prev = a_r - blk.f(b_prev, cos, sin)
            ta, tb = keep[i]
            recon.append(wmax(rel(a_prev, ta), rel(b_prev, tb)))
            a_r, b_r = a_prev, b_prev
    else:
        xm1, xc = torch.zeros_like(e), e
        keep = [(xm1, xc)]
        for blk in m.blocks:
            xm1, xc = xc, xm1 + blk(xc, cos, sin)
            keep.append((xm1, xc))
        xi, xi1 = keep[-1]
        for i, blk in list(enumerate(m.blocks))[::-1]:
            xim1 = xi1 - blk(xi, cos, sin)
            recon.append(rel(xim1, keep[i][0]))
            xi, xi1 = xim1, xi
recon = recon[::-1]

# ---- 2. gradient error vs store-everything reference -------------------------------
m.zero_grad(set_to_none=True)
with autocast():
    loss_rev = m(x, y)
loss_rev.backward()
gr = {k: (p.grad.clone() if p.grad is not None else None) for k, p in m.named_parameters()}
print('no-grad params (reversible path):', [k for k,v in gr.items() if v is None])

m.zero_grad(set_to_none=True)
with autocast():
    e = m.emb(x)
    if args.arch.startswith("rev_euler"):
        a, b = e, e
        for blk in m.blocks:
            a = a + blk.f(b, cos, sin)
            b = b + blk.g(a, cos, sin)
        h = m.ln_f(a + b)
    else:
        xm1, xc = torch.zeros_like(e), e
        for blk in m.blocks:
            xm1, xc = xc, xm1 + blk(xc, cos, sin)
        h = m.ln_f(xm1 + xc)
    loss_ref = lm_loss(h, m.emb.weight, y)
loss_ref.backward()
gf = {k: (p.grad.clone() if p.grad is not None else None) for k, p in m.named_parameters()}
print('no-grad params (reference path):', [k for k,v in gf.items() if v is None])

keys = [k for k in gf if gf[k] is not None and gr.get(k) is not None]
errs = {k: rel(gr[k], gf[k]) for k in keys}
worst = max(errs, key=errs.get)
cos_sims, per_block = [], {}
for k in keys:
    a_, b_ = gr[k].flatten().float(), gf[k].flatten().float()
    c = torch.nn.functional.cosine_similarity(a_, b_, dim=0).item()
    cos_sims.append(c)
    if k.startswith("blocks."):
        per_block.setdefault(f"block{k.split('.')[1]}", []).append(c)
blockwise = {b: round(min(v), 4) for b, v in sorted(per_block.items())}

print(json.dumps({
    "arch": args.arch, "trained_steps": args.steps, "bf16": bool(args.bf16),
    "loss_rev": loss_rev.item(), "loss_ref": loss_ref.item(),
    "recon_rel_err_by_layer": {("x_-1" if i == 0 else f"x_{i-1}"): (round(r, 6) if r is not None else None) for i, r in enumerate(recon)},
    "max_recon_rel_err": wmax(*recon),
    "max_grad_rel_err": errs[worst], "worst_param": worst,
    "min_grad_cosine_sim": min(cos_sims),
    "grad_cosine_by_block": blockwise,
}, indent=2))
