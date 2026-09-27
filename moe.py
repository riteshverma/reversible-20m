"""
Sparse upcycling: turn a trained dense GPTBaseline into a top-k Mixture-of-Experts GPT.

Every block's MLP (fc: d->4d, proj: 4d->d) is copied into E identical experts and a
small random router is added (a zero router would tie every token onto experts 0/1). With top-k gates renormalised to sum to 1, identical
experts make the MoE layer compute *exactly* the dense MLP at conversion time, so the
upcycled model starts from the dense model's loss. Routing then differs per token, the
experts receive different gradients, and they diverge / specialise as training continues.

Router logits are computed in fp32. A Switch-style load-balancing loss
  aux = E * sum_e f_e * P_e   (f_e = fraction of routed slots, P_e = mean router prob)
is stored on the model (model.aux_loss) so evaluate() keeps returning pure CE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GPTBaseline, MLP


class MoEMLP(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2, mult=4):
        super().__init__()
        self.E, self.k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.w_fc = nn.Parameter(torch.empty(n_experts, mult * d_model, d_model))
        self.w_proj = nn.Parameter(torch.empty(n_experts, d_model, mult * d_model))
        self.aux_loss = None
        self.last_load = None            # fraction of routed slots per expert (detached)

    @classmethod
    def from_dense(cls, mlp: MLP, n_experts, top_k, router_std=0.02):
        d = mlp.fc.weight.shape[1]
        m = cls(d, n_experts, top_k, mult=mlp.fc.weight.shape[0] // d).to(mlp.fc.weight.device)
        with torch.no_grad():
            m.w_fc.copy_(mlp.fc.weight.unsqueeze(0).expand_as(m.w_fc))
            m.w_proj.copy_(mlp.proj.weight.unsqueeze(0).expand_as(m.w_proj))
            if router_std > 0:
                nn.init.normal_(m.router.weight, std=router_std)
            else:
                m.router.weight.zero_()
        return m

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        N = xf.shape[0]
        with torch.autocast("cuda", enabled=False):
            logits = F.linear(xf.float(), self.router.weight.float())
            probs = logits.softmax(-1)                                   # (N, E)
        top_p, top_i = probs.topk(self.k, dim=-1)                        # (N, k)
        gates = top_p / top_p.sum(-1, keepdim=True)

        flat_e = top_i.reshape(-1)                                       # (N*k,)
        counts = torch.bincount(flat_e, minlength=self.E)
        load = counts.float() / flat_e.numel()
        self.aux_loss = self.E * (load * probs.mean(0)).sum()
        self.last_load = load.detach()

        order = flat_e.argsort()
        tok = order // self.k                                            # source token per slot
        g = gates.reshape(-1)[order]
        out = torch.zeros(N, d, device=x.device, dtype=torch.float32)
        start = 0
        for e, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            idx = tok[start : start + c]
            h = F.linear(F.gelu(F.linear(xf[idx], self.w_fc[e])), self.w_proj[e])
            out.index_add_(0, idx, h.float() * g[start : start + c, None])
            start += c
        return out.view(B, T, d).to(x.dtype)


class GPTMoE(GPTBaseline):
    """GPTBaseline whose block MLPs are MoEMLPs. forward() returns CE; aux on self.aux_loss."""

    def forward(self, idx, targets=None, ce_chunk=8192):
        out = super().forward(idx, targets, ce_chunk)
        self.aux_loss = sum(b.mlp.aux_loss for b in self.blocks) / len(self.blocks)
        return out

    def expert_load(self):
        return torch.stack([b.mlp.last_load for b in self.blocks])        # (L, E)

    @torch.no_grad()
    def expert_divergence(self):
        """Per layer: mean_e ||W_e - mean(W)|| / ||mean(W)|| over fc+proj (0 at upcycling)."""
        res = []
        for b in self.blocks:
            r = []
            for W in (b.mlp.w_fc, b.mlp.w_proj):
                mu = W.mean(0, keepdim=True)
                r.append(((W - mu).flatten(1).norm(dim=1) / mu.norm()).mean().item())
            res.append(sum(r) / 2)
        return res


def upcycle(dense: GPTBaseline, n_experts=8, top_k=2, router_std=0.02) -> GPTMoE:
    """Returns a new GPTMoE sharing no storage with `dense`."""
    import copy
    moe = copy.deepcopy(dense)
    moe.__class__ = GPTMoE
    for blk in moe.blocks:
        blk.mlp = MoEMLP.from_dense(blk.mlp, n_experts, top_k, router_std)
    moe.aux_loss = None
    return moe
