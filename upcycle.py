"""Dense -> MoE sparse-upcycling experiment.

One LR schedule (warmup + cosine to 0.1x) spans the whole --total-tokens budget.

  phase 1  dense GPTBaseline trained from scratch up to --stop-tokens, checkpointed
           (model, AdamW state, batch-sampler RNG, schedule position).
  phase 2  resume from that checkpoint for the rest of the schedule, either
             --mode dense : control, the same dense model keeps training
             --mode moe   : every MLP upcycled to E experts / top-k (AdamW moments copied
                            into each expert, fresh router), then trained
           Both arms see exactly the same batches and learning rates.

  python upcycle.py dense --stop-tokens 20e6 --out runs/up_dense20M
  python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode dense --out runs/up_dense_cont
  python upcycle.py resume --init runs/up_dense20M/ckpt.pt --mode moe --experts 8 --topk 2 --out runs/up_moe8x2
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPTBaseline, count_params
from moe import GPTMoE, upcycle
from train import DATA, VOCAB, evaluate, get_batch


def lr_at(step, steps, warmup, peak):
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))


def param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append((name, p))
    return decay, no_decay


def make_opt(model, lr):
    decay, no_decay = param_groups(model)
    opt = torch.optim.AdamW(
        [{"params": [p for _, p in decay], "weight_decay": 0.1},
         {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    return opt, [n for n, _ in decay] + [n for n, _ in no_decay]


def transfer_opt_state(opt, moe, dense_names, dense_opt_sd, n_experts):
    """Seed the MoE optimizer with the dense AdamW moments. Expert tensors get the dense
    MLP's moments replicated E times; the router starts with no state."""
    src = {n: dense_opt_sd["state"][i] for i, n in enumerate(dense_names) if i in dense_opt_sd["state"]}
    copied, fresh = 0, []
    for name, p in moe.named_parameters():
        s_name = name.replace("mlp.w_fc", "mlp.fc.weight").replace("mlp.w_proj", "mlp.proj.weight")
        if s_name not in src:
            fresh.append(name)
            continue
        s = src[s_name]
        st = {}
        for k, v in s.items():
            if torch.is_tensor(v) and v.ndim > 0 and v.shape != p.shape:
                v = v.unsqueeze(0).expand(n_experts, *v.shape)
            st[k] = v.to(p.device).contiguous() if torch.is_tensor(v) else v
        opt.state[p] = st
        copied += 1
    return copied, fresh


def train_segment(model, opt, train_mm, val_mm, g, args, start, end, steps, warmup, log):
    is_moe = isinstance(model, GPTMoE)
    n_train = len(train_mm) - args.seq - 1
    curve, evals, ema = [], [], None
    t0, tok_t0 = time.time(), None
    for step in range(start, end):
        lr = lr_at(step, steps, warmup, args.lr)
        for grp in opt.param_groups:
            grp["lr"] = lr
        idxs = torch.randint(0, n_train, (args.batch,), generator=g).numpy()
        x, y = get_batch(train_mm, idxs, args.seq, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce = model(x, y)
        aux = model.aux_loss if is_moe else None
        loss = ce + args.aux_coef * aux if is_moe else ce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        cv = ce.item()
        ema = cv if ema is None else 0.95 * ema + 0.05 * cv
        curve.append((step, cv, aux.item() if is_moe else 0.0, lr))
        if step == start + 5:
            torch.cuda.synchronize()
            tok_t0 = time.time()
        if (step + 1) % args.log_every == 0 or step == end - 1:
            tps = (step - start - 5) * args.batch * args.seq / (time.time() - tok_t0) if tok_t0 else 0
            msg = (f"step {step+1}/{steps} ce {cv:.4f} ema {ema:.4f} lr {lr:.2e} "
                   f"tok/s {tps/1e3:.1f}k elapsed {time.time()-t0:.0f}s")
            if is_moe:
                ld = model.expert_load()
                msg += (f" aux {aux.item():.3f} max-load/layer "
                        + " ".join(f"{v:.2f}" for v in ld.max(1).values.tolist()))
            log(msg)
        if (step + 1) % args.eval_every == 0 or step == end - 1:
            v = evaluate(model, val_mm, args.seq, "cuda", max_tokens=args.eval_tokens)
            evals.append((step + 1, v))
            log(f"  eval step {step+1} tokens {(step+1)*args.batch*args.seq/1e6:.1f}M val {v:.4f}")
    wall = time.time() - t0
    tps = (end - start - 5) * args.batch * args.seq / (time.time() - tok_t0) if tok_t0 else 0
    return curve, evals, ema, wall, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dense", "resume"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--mode", choices=["dense", "moe"], default="dense")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--router-std", type=float, default=0.02)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--total-tokens", type=float, default=50e6)
    ap.add_argument("--stop-tokens", type=float, default=20e6)
    ap.add_argument("--lr", type=float, default=7.5e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-tokens", type=int, default=500_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "stdout.log", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    train_mm = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val_mm = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    steps = int(args.total_tokens // (args.batch * args.seq))
    warmup = max(30, int(steps * args.warmup_frac))
    info = {}

    if args.phase == "dense":
        torch.manual_seed(args.seed)
        model = GPTBaseline(VOCAB, seq=args.seq).cuda()
        opt, names = make_opt(model, args.lr)
        g = torch.Generator().manual_seed(args.seed)
        start, end = 0, int(args.stop_tokens // (args.batch * args.seq))
        arm = "dense_phase1"
    else:
        # load on CPU: the MoE needs most of a 6 GB card, so the checkpoint must not
        # stay resident on the GPU for the whole run
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        torch.manual_seed(args.seed + 1)
        dense = GPTBaseline(VOCAB, seq=args.seq).cuda()
        dense.load_state_dict(ck["model"])
        g = torch.Generator()
        g.set_state(ck["gen_state"].cpu())
        start, end = ck["step"], steps
        assert ck["steps_total"] == steps, "resume must use the same --total-tokens/--batch"
        v_dense = evaluate(dense, val_mm, args.seq, "cuda", max_tokens=args.eval_tokens)
        if args.mode == "dense":
            model = dense
            opt, names = make_opt(model, args.lr)
            opt.load_state_dict(ck["opt"])
            arm = "dense_continue"
        else:
            model = upcycle(dense, args.experts, args.topk, args.router_std)
            del dense
            opt, names = make_opt(model, args.lr)
            copied, fresh = transfer_opt_state(opt, model, ck["opt_names"], ck["opt"], args.experts)
            log(f"optimizer state copied for {copied} tensors; fresh: {fresh}")
            arm = f"moe_{args.experts}x_top{args.topk}"
        del ck
        torch.cuda.empty_cache()
        v0 = evaluate(model, val_mm, args.seq, "cuda", max_tokens=args.eval_tokens)
        info = {"val_at_resume_dense_ckpt": v_dense, "val_at_resume_this_model": v0}
        log(f"resume at step {start}: dense ckpt val {v_dense:.5f} | {arm} val {v0:.5f} "
            f"(delta {v0 - v_dense:+.2e})")

    pc = count_params(model)
    if isinstance(model, GPTMoE):
        per_expert = model.blocks[0].mlp.w_fc[0].numel() + model.blocks[0].mlp.w_proj[0].numel()
        inactive = len(model.blocks) * (args.experts - args.topk) * per_expert
        pc["active_per_token"] = pc["total"] - inactive
    log(f"arm={arm} params={pc} steps {start}->{end} of {steps} lr={args.lr}")

    curve, evals, ema, wall, tps = train_segment(
        model, opt, train_mm, val_mm, g, args, start, end, steps, warmup, log)
    if info:
        evals = [(start, info["val_at_resume_this_model"])] + evals

    val_full = evaluate(model, val_mm, args.seq, "cuda")          # 2M-token val, as train.py
    metrics = {
        "arm": arm, "params": pc, "batch": args.batch, "seq": args.seq, "lr": args.lr,
        "start_step": start, "end_step": end, "steps_total": steps,
        "tokens_start": start * args.batch * args.seq, "tokens_end": end * args.batch * args.seq,
        "final_train_ce_ema": ema, "final_val_loss_2M": val_full, "val_ppl": math.exp(val_full),
        "tokens_per_s": tps, "wall_s": wall,
        "peak_mem_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
        "gpu": torch.cuda.get_device_name(0), **info,
    }
    if isinstance(model, GPTMoE):
        metrics.update(experts=args.experts, topk=args.topk, aux_coef=args.aux_coef,
                       router_std=args.router_std,
                       expert_divergence_per_layer=model.expert_divergence(),
                       final_expert_load=model.expert_load().tolist())
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / "curve.npy", np.array(curve))
    np.save(out / "evals.npy", np.array(evals))
    if args.phase == "dense":
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "opt_names": names,
                    "gen_state": g.get_state(), "step": end, "steps_total": steps}, out / "ckpt.pt")
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
