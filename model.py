"""
20M-class GPT with three activation-memory strategies, all with matched parameter counts:

  - "baseline":  standard pre-norm transformer, autograd stores every activation.
  - "rev_euler": two-stream reversible network (Gomez et al. 2017). The coupled update
                 (a,b) -> (a + f(b), b + g(a + f(b))) is a semi-implicit *Euler* step of a
                 2D ODE; its inverse is exact, so only the final pair is stored.
  - "rev_mid":   single-stream *explicit midpoint* (leapfrog / Verlet) reversible stack:
                 x_{i+1} = x_{i-1} + f_i(x_i). Symmetric, 2nd-order; exact inverse
                 x_{i-1} = x_{i+1} - f_i(x_i) from adjacent activation pairs. Full-width
                 blocks (no channel halving).

All variants: d_model=512, 6 layers (blocks), 8 heads, MLP 4x, RoPE, tied embeddings,
vocab 8192 -> 18.87M non-embedding params + 4.19M embedding = 23.06M total.

Streams/residual are kept in fp32; matmuns run under bf16 autocast.
A fused chunked cross-entropy keeps logits memory O(chunk) in fwd and bwd.
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import custom_fwd, custom_bwd



# ------------------------------------------------------------------- autocast plumbing
# A reversible backward recomputes each block inside torch.autograd.grad. Under autocast
# the bf16 copy of every weight is CACHED, and because the forward ran under no_grad the
# cached copy carries no grad_fn -- the recompute then cannot reach the fp32 parameter and
# autograd.grad(..., allow_unused=True) silently returns None for every Linear weight.
# The model still trains (LayerNorm gains and the embedding are reached directly), so the
# failure is invisible in the loss curve. Re-entering autocast with cache_enabled=False
# inside the backward forces a fresh, differentiable cast.

def _amp_state():
    return (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))


def _amp_ctx(state):
    enabled, dtype = state
    return torch.autocast("cuda", dtype=dtype, enabled=enabled, cache_enabled=False)


def _require_all_grads(grads, params):
    if any(g is None for g in grads):
        missing = sum(g is None for g in grads)
        raise RuntimeError(
            f"reversible backward produced no gradient for {missing}/{len(params)} block "
            "parameters; the autocast weight cache has detached them from the graph")


# ----------------------------------------------------------------------------- blocks

def rope_cos_sin(T, head_dim, device, base=10000.0):
    half = head_dim // 2
    inv_freq = base ** (-torch.arange(half, device=device, dtype=torch.float32) / half)
    t = torch.arange(T, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                      # (T, half)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, T, H, D). Rotate-half convention on the last dim.
    D = x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h = n_heads
        self.hd = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3 * self.h, self.hd).chunk(3, dim=2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # (B, H, T, hd)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.fc = nn.Linear(d_model, mult * d_model, bias=False)
        self.proj = nn.Linear(mult * d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Standard pre-norm block (used by baseline and rev_mid)."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class RevBlock(nn.Module):
    """Two-stream reversible block: f = attention on stream b, g = MLP on stream a."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=False)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model, bias=False)
        self.mlp = MLP(d_model)

    def f(self, b, cos, sin):
        return self.attn(self.ln1(b), cos, sin)

    def g(self, a, cos, sin):
        return self.mlp(self.ln2(a))


# ------------------------------------------------------------- fused chunked cross-entropy

class FusedCE(torch.autograd.Function):
    """Chunked cross-entropy over flattened tokens; logits never stored for backward.
    forward: loss = mean CE. backward: dlogits = (softmax - onehot) * scale per chunk,
    accumulated into dh and dW with bf16 matmuls."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, h, weight, targets, chunk=8192):
        # h: (N, d) fp32, weight: (V, d), targets: (N,) int64
        ctx.chunk = chunk
        ctx.save_for_backward(h, weight, targets)
        N = h.shape[0]
        with torch.no_grad():
            loss = h.new_zeros((), dtype=torch.float32)
            for i in range(0, N, chunk):
                hc, tc = h[i : i + chunk], targets[i : i + chunk]
                logits = F.linear(hc, weight).float()
                loss = loss + F.cross_entropy(logits, tc, reduction="sum")
        return loss / N

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        h, weight, targets = ctx.saved_tensors
        N, d = h.shape
        chunk = ctx.chunk
        dh = torch.zeros_like(h)
        dW = torch.zeros_like(weight)
        scale = grad_out / N
        for i in range(0, N, chunk):
            hc, tc = h[i : i + chunk], targets[i : i + chunk]
            logits = F.linear(hc, weight).float()
            p = torch.softmax(logits, dim=-1)
            p[torch.arange(tc.shape[0], device=h.device), tc] -= 1.0
            p = p * scale
            dW += p.T @ hc                       # TF32 matmul (enabled in train.py)
            dh[i : i + chunk] = p @ weight
        return dh, dW, None, None


def lm_loss(h, emb_weight, targets, chunk=8192):
    """h: (B, T, d) -> scalar CE loss."""
    B, T, d = h.shape
    return FusedCE.apply(h.reshape(B * T, d), emb_weight, targets.reshape(B * T), chunk)


# ------------------------------------------------------------------------- reversible stacks

class RevEulerFn(torch.autograd.Function):
    """Two-stream reversible stack. Stores only the final (a, b); backward reconstructs
    the chain top-down and recomputes each block with grad to produce exact gradients."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, a0, b0, blocks, cos, sin, exact_seed=False):
        ctx.blocks = blocks
        ctx.cos, ctx.sin = cos, sin
        ctx.amp = _amp_state()
        ctx.exact_seed = exact_seed
        ctx.seed = (a0, b0) if exact_seed else None
        with torch.no_grad():
            a, b = a0, b0
            for blk in blocks:
                a = a + blk.f(b, cos, sin)
                b = b + blk.g(a, cos, sin)   # g sees the UPDATED a (sequential coupling -> invertible)
        ctx.save_for_backward(a, b)
        return a, b

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, da, db):
        blocks, cos, sin = ctx.blocks, ctx.cos, ctx.sin
        a, b = ctx.saved_tensors
        for i, blk in list(enumerate(blocks))[::-1]:
            if ctx.exact_seed and i == 0:
                a_prev, b_prev = ctx.seed          # exact (a_0, b_0); no cancellation
            else:
                with _amp_ctx(ctx.amp), torch.no_grad():
                    b_prev = b - blk.g(a, cos, sin)
                    a_prev = a - blk.f(b_prev, cos, sin)
            a_prev = a_prev.detach().requires_grad_(True)
            b_prev = b_prev.detach().requires_grad_(True)
            params = [p for p in blk.parameters() if p.requires_grad]
            with _amp_ctx(ctx.amp), torch.enable_grad():
                a_new = a_prev + blk.f(b_prev, cos, sin)
                b_new = b_prev + blk.g(a_new, cos, sin)
            grads = torch.autograd.grad(
                (a_new, b_new), [a_prev, b_prev] + params,
                grad_outputs=(da, db), allow_unused=True,
            )
            _require_all_grads(grads[2:], params)
            da, db = grads[0], grads[1]
            for p, g in zip(params, grads[2:]):
                if g is not None:
                    p.grad = g if p.grad is None else p.grad + g
            a, b = a_prev.detach(), b_prev.detach()
        return da, db, None, None, None, None


class LeapfrogFn(torch.autograd.Function):
    """Single-stream explicit-midpoint (leapfrog) reversible stack:
    x_{i+1} = x_{i-1} + f_i(x_i), x_{-1} = 0. Stores only (x_{N-1}, x_N)."""

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, x0, blocks, cos, sin, exact_seed=False):
        ctx.blocks = blocks
        ctx.cos, ctx.sin = cos, sin
        ctx.amp = _amp_state()
        ctx.exact_seed = exact_seed
        with torch.no_grad():
            xm1 = torch.zeros_like(x0)
            x = x0
            for blk in blocks:
                xm1, x = x, xm1 + blk(x, cos, sin)
        ctx.save_for_backward(*((xm1, x, x0) if exact_seed else (xm1, x)))
        return xm1, x

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, dxm1_top, dx_top):
        blocks, cos, sin = ctx.blocks, ctx.cos, ctx.sin
        saved = ctx.saved_tensors            # (x_{N-1}, x_N[, x_0])
        xi, xi1 = saved[0], saved[1]
        x0_exact = saved[2] if ctx.exact_seed else None
        gxi, gxi1 = dxm1_top, dx_top         # their total grads from above
        for i, blk in list(enumerate(blocks))[::-1]:
            if x0_exact is not None and i == 0:
                # Recovering x_0 by subtraction is catastrophically ill-conditioned: it is
                # embedding-scale while x_1..x_N carry the accumulated residual. x_0 is an
                # input to this Function, so use it directly and skip the subtraction.
                xi = x0_exact
                xim1 = torch.zeros_like(xi)
            else:
                with _amp_ctx(ctx.amp), torch.no_grad():
                    xim1 = xi1 - blk(xi, cos, sin)   # x_{i-1} = x_{i+1} - f_i(x_i)
            xim1 = xim1.detach().requires_grad_(True)
            xi_leaf = xi.detach().requires_grad_(True)
            params = [p for p in blk.parameters() if p.requires_grad]
            with _amp_ctx(ctx.amp), torch.enable_grad():
                xi1_new = xim1 + blk(xi_leaf, cos, sin)
            grads = torch.autograd.grad(
                (xi1_new,), [xim1, xi_leaf] + params,
                grad_outputs=(gxi1,), allow_unused=True,
            )
            _require_all_grads(grads[2:], params)
            # grads[0]: identity path to x_{i-1}; grads[1]: path through f_i into x_i
            new_gxim1 = grads[0]
            new_gxi = gxi + grads[1]
            for p, g in zip(params, grads[2:]):
                if g is not None:
                    p.grad = g if p.grad is None else p.grad + g
            xi, xi1 = xim1.detach(), xi.detach()
            gxi, gxi1 = new_gxim1, new_gxi
        # loop ends with (x_{-1}, x_0): gradient w.r.t. x0 is gxi1; x_{-1} was a constant zero
        return gxi1, None, None, None, None


# --------------------------------------------------------------------------------- models

class GPTBaseline(nn.Module):
    def __init__(self, vocab, d_model=512, n_layers=6, n_heads=8, seq=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.seq = seq
        cos, sin = rope_cos_sin(seq, d_model // n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        # scale residual output projections for stable depth
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(blk.mlp.proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, ce_chunk=8192):
        cos = self.cos.to(idx.device)[: idx.shape[1]]
        sin = self.sin.to(idx.device)[: idx.shape[1]]
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        if targets is None:
            return x
        return lm_loss(x, self.emb.weight, targets, ce_chunk)


class GPTRevEuler(nn.Module):
    """Two-stream reversible GPT. Each stream is d_model wide; params per block match baseline."""
    exact_seed = False

    def __init__(self, vocab, d_model=512, n_layers=6, n_heads=8, seq=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([RevBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.seq = seq
        cos, sin = rope_cos_sin(seq, d_model // n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(GPTBaseline._init)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(blk.mlp.proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx, targets=None, ce_chunk=8192):
        cos = self.cos.to(idx.device)[: idx.shape[1]]
        sin = self.sin.to(idx.device)[: idx.shape[1]]
        e = self.emb(idx)
        a, b = RevEulerFn.apply(e, e, self.blocks, cos, sin, self.exact_seed)
        x = self.ln_f(a + b)
        if targets is None:
            return x
        return lm_loss(x, self.emb.weight, targets, ce_chunk)


class GPTRevMid(nn.Module):
    """Single-stream leapfrog (explicit midpoint) reversible GPT; blocks identical to baseline."""
    exact_seed = False

    def __init__(self, vocab, d_model=512, n_layers=6, n_heads=8, seq=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, bias=False)
        self.seq = seq
        cos, sin = rope_cos_sin(seq, d_model // n_heads, "cpu")
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(GPTBaseline._init)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02 / math.sqrt(2 * n_layers))
            nn.init.normal_(blk.mlp.proj.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, idx, targets=None, ce_chunk=8192):
        cos = self.cos.to(idx.device)[: idx.shape[1]]
        sin = self.sin.to(idx.device)[: idx.shape[1]]
        e = self.emb(idx)
        x_prev, x = LeapfrogFn.apply(e, self.blocks, cos, sin, self.exact_seed)
        out = self.ln_f(x_prev + x)
        if targets is None:
            return out
        return lm_loss(out, self.emb.weight, targets, ce_chunk)


class GPTRevMidSeed(GPTRevMid):
    """rev_mid, but the backward takes x_0 from the saved input instead of reconstructing
    it. Costs one extra stored activation; removes the block-0 gradient corruption."""
    exact_seed = True


class GPTRevEulerSeed(GPTRevEuler):
    """rev_euler with the same exact-seed treatment."""
    exact_seed = True


ARCHS = {"baseline": GPTBaseline, "rev_euler": GPTRevEuler, "rev_mid": GPTRevMid,
         "rev_mid_seed": GPTRevMidSeed, "rev_euler_seed": GPTRevEulerSeed}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    emb = model.emb.weight.numel()
    return {"total": total, "embedding": emb, "non_embedding": total - emb}
