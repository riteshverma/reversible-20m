"""
Gradient-correctness verification: custom reversible Functions and FusedCE must produce
the same gradients as a naive store-everything autograd reference. Run in fp32 on CPU
(SDPA works on CPU too) so any mismatch is numerical, not hardware.
"""

import torch
import torch.nn.functional as F

from model import GPTBaseline, GPTRevEuler, GPTRevMid, FusedCE, lm_loss

torch.manual_seed(0)
V, D, L, H, T, B = 256, 64, 4, 4, 32, 2


def copy_state(src, dst):
    dst.load_state_dict(src.state_dict())


def param_grads(model):
    return {k: p.grad.clone() for k, p in model.named_parameters()}


def rel_err(a, b):
    denom = b.abs().max().clamp_min(1e-12)
    return (a - b).abs().max() / denom


def check_reversible(RevCls, name, top="sum"):
    base = GPTBaseline(V, D, L, H, T)
    rev = RevCls(V, D, L, H, T)
    copy_state(base, rev)

    idx = torch.randint(0, V, (B, T))
    tgt = torch.randint(0, V, (B, T))

    # reference: same architecture run with plain autograd (store everything)
    ref = RevCls(V, D, L, H, T)
    copy_state(base, ref)
    x = ref.emb(idx)
    cos, sin = ref.cos, ref.sin
    if name == "rev_euler":
        a, b = x, x
        for blk in ref.blocks:
            a = a + blk.f(b, cos, sin)
            b = b + blk.g(a, cos, sin)
        h = ref.ln_f(a + b)
    else:  # leapfrog
        xm1 = torch.zeros_like(x)
        for blk in ref.blocks:
            xm1, x = x, xm1 + blk(x, cos, sin)
        h = ref.ln_f(xm1 + x)
    h_ref = ref.ln_f(a + b) if name == "rev_euler" else None
    if name != "rev_euler":
        h_ref = h
    logits_ref = F.linear(h_ref.reshape(-1, D), ref.emb.weight).float()
    loss_ref = F.cross_entropy(logits_ref, tgt.reshape(-1))
    loss_ref.backward()
    ref_grads = param_grads(ref)

    # custom reversible path
    rev.zero_grad(set_to_none=True)
    loss_rev = rev(idx, tgt)
    loss_rev.backward()
    rev_grads = param_grads(rev)

    worst = 0.0
    for k in ref_grads:
        e = rel_err(rev_grads[k], ref_grads[k])
        worst = max(worst, e.item())
    print(f"[{name}] loss_ref={loss_ref.item():.6f} loss_rev={loss_rev.item():.6f} "
          f"d_loss={abs(loss_ref - loss_rev).item():.2e}  max_rel_grad_err={worst:.2e}")
    assert worst < 1e-5, f"{name}: gradient mismatch {worst}"
    return worst


def check_fused_ce():
    torch.manual_seed(1)
    N, d, V2 = 1000, 64, 512
    h = torch.randn(N, d, requires_grad=True)
    w = torch.randn(V2, d, requires_grad=True)
    t = torch.randint(0, V2, (N,))

    # reference
    h2 = h.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    loss_ref = F.cross_entropy(F.linear(h2, w2).float(), t)
    loss_ref.backward()

    loss_fused = FusedCE.apply(h, w, t, 256)
    loss_fused.backward()

    dl = abs(loss_ref.item() - loss_fused.item()) / loss_ref.item()
    eh = rel_err(h.grad, h2.grad).item()
    ew = rel_err(w.grad, w2.grad).item()
    print(f"[FusedCE] rel_loss_err={dl:.2e} dH_err={eh:.2e} dW_err={ew:.2e}")
    assert dl < 1e-5 and eh < 1e-4 and ew < 1e-4


if __name__ == "__main__":
    check_fused_ce()
    check_reversible(GPTRevEuler, "rev_euler")
    check_reversible(GPTRevMid, "rev_mid")
    print("ALL GRADIENT CHECKS PASSED")
