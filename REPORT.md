# Reversible 20M-parameter LLM: 50M-token training report

Hardware: **NVIDIA RTX 3060 Laptop, 6 GB** (5120 MB actually free; Windows holds ~1 GB).
PyTorch 2.10.0+cu126, bf16 autocast, fused AdamW, cosine schedule decaying to 10% of peak,
grad-clip 1.0, seed 1234. Colab was not used — everything ran locally on this GPU.

**Model** (identical parameter count across all variants): d_model 512, 6 blocks, 8 heads,
MLP 4x, RoPE, tied embeddings, vocab 8192, seq 512.
**23.08M params total** = 18.88M non-embedding + 4.19M embedding.

**Data**: one FineWeb-Edu parquet shard, byte-level BPE trained to 8192 tokens,
52M train / 5M val tokens as uint16 memmaps.

Memory is reported as `torch.cuda.max_memory_allocated` (tensor bytes) and
`max_memory_reserved` (what the caching allocator took from the driver — this is the number
that decides whether a batch actually fits).

---

## 1. The three required runs

| # | run | arch | batch | steps | LR | **val loss** | ppl | **tok/s** | **peak alloc** | peak reserved | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline, fixed batch | standard pre-norm | 16 | 6103 | 7.5e-4 | **3.783** | 43.9 | 40.5k | **2130 MB** | 2348 MB | 20.6 min |
| 2 | + reversibility, same batch | `rev_mid` (midpoint) | 16 | 6103 | 7.5e-4 | **3.797** | 44.6 | 28.8k | **1157 MB** | 1482 MB | 28.9 min |
| 3 | + reversibility, max batch | `rev_mid` (midpoint) | **192** | 508 | 1.3e-3 | **4.254** | 70.4 | 35.2k | **4502 MB** | 5118 MB | 23.6 min |

All three consumed ~50M tokens (49.9–50.0M).

### Headline numbers

- **Loss**: reversibility at the same batch is a **wash** — 3.797 vs 3.783, a 0.014-nat gap
  (1.4% perplexity). Matched parameter count, matched token budget, each at its own tuned LR.
- **Memory**: **1.84x less** peak allocated (1157 vs 2130 MB). Counting only activations
  (subtracting the fixed 369 MB of params + grads + Adam moments) it is **2.2x less**
  (788 vs 1761 MB) — reversibility only touches activation memory.
- **Speed**: reversibility costs **~1.56x** in compute (see §4 — the raw tok/s column above is
  not a fair comparison on this GPU).
- **Max batch**: reversibility raises the largest batch that fits 6 GB from **56 to 192 (3.4x)**.

---

## 2. Which reversible variant worked: **midpoint (leapfrog)**

Both were implemented with matched parameter counts:

- `rev_euler` — two-stream RevNet/Reformer coupling, `a += f(b); b += g(a)`. This is a
  semi-implicit **Euler** step; attention only ever reads stream `b`, the MLP only reads `a`.
- `rev_mid` — single-stream **explicit midpoint / leapfrog**, `x_{i+1} = x_{i-1} + f_i(x_i)`,
  `x_{-1} = 0`. Symmetric, 2nd-order, and it reuses *the same full-width pre-norm block as the
  baseline*.

Selection grid, 10M tokens, batch 16, val loss. Every architecture is bracketed on both sides of
its optimum, so this is best-vs-best rather than best-vs-mistuned:

| LR | baseline | `rev_mid` | `rev_euler` |
|---|---|---|---|
| 2e-4 | 5.169 | — | 5.147 |
| 4e-4 | 4.912 | 4.853 | **4.921** |
| **7.5e-4** | **4.754** | **4.672** | 4.980 |
| 1.5e-3 | 4.952 | 4.846 | 5.081 |
| 3e-3 | 5.181 | — | 5.269 |
| 6e-3 | — | — | 5.894 |

**Midpoint wins clearly**: 4.672 vs 4.921 for Euler — and it is the only variant that beats the
baseline (4.754) at this horizon. The reason is likely structural rather than numerical: the
leapfrog stack reuses the baseline's block verbatim and only changes where the skip connection
lands, whereas the two-stream Euler coupling restricts attention to read a stream that
accumulates only MLP outputs. Midpoint was used for runs 2 and 3.

---

## 3. Pushing to the maximum batch size

A batch "fits" only if **peak reserved < 5120 MB free**. Exceeding it does not OOM on Windows —
the driver silently spills into shared system memory and throughput collapses, which is a worse
failure than a crash because it looks like it worked.

| arch | max batch that fits | peak reserved at that batch | first batch that spills |
|---|---|---|---|
| baseline | **56** | 4836 MB | 64 (5300 MB) |
| `rev_euler` | **192** | 4878 MB | 208 (5254 MB) |
| `rev_mid` | **192** | 5118 MB | 208 (5326 MB) |

Reversibility buys a **3.4x larger batch** (56 → 192) on the same card.

**But at a fixed 50M-token budget, the maximum batch is the wrong choice.** Batch 192 means only
**508 optimizer steps** versus 6103, and the loss is much worse — 4.254 vs 3.797, a 0.46-nat
regression. The large-batch LR was swept and properly bracketed, so this is not a tuning artifact:

| batch 192, LR | 6.5e-4 | **1.3e-3** | 2.6e-3 | 5.2e-3 |
|---|---|---|---|---|
| val loss | 4.469 | **4.254** | 4.402 | 4.989 |

The optimum sits at only ~1.7x the batch-16 LR even though the batch is 12x larger —
square-root scaling (which would predict 2.6e-3) overshoots, because at 508 steps the run is
nowhere near convergence and a large LR mostly costs early stability.

Max batch is worth having when memory is the binding constraint (longer sequences, a bigger
model, avoiding gradient accumulation) — not as a way to spend a fixed token budget faster.

---

## 4. Speed, and why the per-run tok/s column is misleading

This is a **laptop** 3060. Under sustained load it heats to 88–89 °C and throttles from
**1882 MHz to ~800–1000 MHz**, roughly halving throughput mid-run:

| phase | SM clock | power |
|---|---|---|
| first ~100 s | 1640–1880 MHz | ~115 W |
| sustained training | ~800–1000 MHz (noisy) | ~57–60 W |
| dense matmul stream | 1387 MHz (stable) | ~71 W |

The limiter is the driver's **SW thermal slowdown** governor (`clocks_event_reasons.sw_thermal_slowdown=Active`;
HW thermal and power cap both inactive) — it is workload-pattern sensitive, which is why a dense
matmul stream holds 1387 MHz while the burstier training loop settles lower.

So whichever config happened to start on a cooler card looks faster, and identical configs
measured anywhere from 28.3k to 40.5k tok/s across runs. Two fixes, both in the repo:

1. **`thermal_bench.py`** — heat-soaks the GPU with a matmul burn until temperature plateaus
   (88 °C, ~4 min), *then* runs the interleaved bursts with continuous SM-clock telemetry.
   Round-to-round spread collapses from ~8% to **<0.5%**.
2. **tok/s per MHz** — clock-normalized throughput, invariant to whatever the governor is doing.

Steady-state results (batch 16, pinned by soak, 1387 MHz throughout):

| arch | tok/s @1387 MHz | tok/s per MHz | vs baseline |
|---|---|---|---|
| baseline | **63.7k** | 45.9 | 1.00x |
| `rev_mid` | **41.7k** | 30.1 | **1.53x slower** |
| `rev_euler` | **42.5k** | 30.7 | **1.50x slower** |

**Reversibility costs ~1.5x wall-clock** — more than the "~33% extra forward" the theory
suggests, because the backward re-runs each block under `torch.autograd.grad` with a fresh
autocast cast per block (see §5a), which does not fuse as well as a plain backward.

### 4a. Independent cross-check: the boost-regime thermal guard

The soak method above measures everything in the *throttled* regime. `repeat_guarded.py` +
`train.py --thermal-guard` attack the same confound from the opposite end: start from a cool
card (<=65 C) and train only while the GPU is still boosting, stopping the moment the governor
engages (`sw_thermal_slowdown` Active, or >=87 C, two consecutive 2 s samples).

Verified before use — the guard tripped at 87 C / 1762 MHz while still at 114 W, i.e. at
throttle *onset*, before any clock collapse:

| t | temp | SM clock | power | sw_thermal |
|---|---|---|---|---|
| 0 s | 62 C | 1702 MHz | 28 W | False |
| 24 s | 83 C | 1800 MHz | 115 W | False |
| 43 s | 87 C | 1762 MHz | 112 W | **True -> stop** |

Unthrottled throughput for the three main configs (min clock 1702 MHz throughout, so entirely
inside the boost regime):

| run | boost tok/s | throttled tok/s (§1) | gain | tokens before trip |
|---|---|---|---|---|
| baseline b=16 | **74.8k** | 40.5k | 1.85x | 1.96M (3.9% of 50M) |
| `rev_mid` b=16 | **49.2k** | 28.8k | 1.71x | 1.09M (2.2%) |
| `rev_mid` b=192 | **61.5k** | 35.2k | 1.75x | 1.18M (2.4%) |

**The two methods agree to within 1%** on the quantity that matters:

| method | regime | baseline / `rev_mid` |
|---|---|---|
| interleaved heat-soak (`thermal_bench.py`) | throttled, 1387 MHz | 1.53x |
| boost-regime guard (`repeat_guarded.py`) | unthrottled, >=1702 MHz | **1.52x** |

So the ~1.5x cost of reversibility is a property of the code, not of when a run happened to
start. The batch-192 speedup reproduces too (1.25x over b=16 in boost vs 1.22x throttled).

**What this also proves: the 50M-token experiment cannot be run unthrottled on this machine.**
The boost regime lasts **22-27 seconds** — 2-4% of one run. Sustaining it means dissipating
114 W indefinitely against the ~59 W this chassis holds at its 89 C ceiling, a ~2x cooling
deficit that a cooling pad's typical 3-8 C does not bridge. The throttled numbers in §1 are the
real sustained performance of this hardware; the guard is a throughput probe, not a training
mode, and its loss values (EMA 5.8-8.7 at 1-2M tokens) are meaningless. Caveat: these runs are
12-239 steps, so tok/s carries some warm-up amortization — the b=192 entry timed only 7 steps
and is the noisiest. Note also that this is a *measurement-regime* guard, not a hardware safety
device; the card's own protection sits near 93 C and was never approached.

For deterministic clocks the lock needs an **admin** shell (a normal one is refused):
`nvidia-smi -lgc 1000,1000` to pin, `nvidia-smi -rgc` to restore — MSI Afterburner works too.
Beyond that it is physics: cooling pad / elevated rear vents, clean fans, low ambient
temperature, AC plugged in (the 115 W cap is only available on AC).

---

## 5. Other findings

### 5a. A silent, catastrophic bug: the autocast weight cache kills reversible gradients

**This invalidated an entire first round of results.** Under `torch.autocast`, the bf16 copy of
each weight is **cached**. A reversible backward recomputes blocks inside
`torch.autograd.grad`, but the forward ran under `torch.no_grad()` — so the cached bf16 copy
carries **no `grad_fn`**, the recompute cannot reach the fp32 parameter, and
`allow_unused=True` returns `None` for **every Linear weight in every block**.

The result: all attention and MLP matrices were **frozen at initialization**. Only LayerNorm
gains and the embedding trained. The loss still fell from 7.5 to 4.9, so nothing looked wrong.

`gradcheck.py` passed at 6e-7 relative error the whole time — because it runs **fp32 on CPU with
no autocast**, where there is no weight cache. Correctness tests must run in the precision and on
the device you actually train in.

The fix is one context manager — re-enter autocast with the cache disabled inside the backward:

```python
with torch.autocast("cuda", dtype=dtype, cache_enabled=False), torch.enable_grad():
    ...
```

Impact, same config (10M tokens, batch 16), before vs after:

| | best val | best LR | LR response |
|---|---|---|---|
| broken | 5.267 | 6e-3 | nearly flat over a 4x LR range |
| fixed | **4.672** | 7.5e-4 | sharp, clean interior minimum |

**A flat LR response was the tell.** A model that barely cares about its learning rate over a 4x
range is usually not training. `model.py` now raises rather than silently accepting a `None`
gradient for any block parameter.

This flipped the variant verdict too: while broken, `rev_euler` appeared to beat `rev_mid`
(5.267 vs 5.280); with real gradients the ordering reverses decisively (4.672 vs 4.980).

### 5b. Reconstruction of the first activation is catastrophically ill-conditioned

Measured against a store-everything autograd reference **on GPU, in bf16, with trained weights**
(not at init) — gradient cosine similarity per block for `rev_mid`:

| block | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| cosine vs true grad | **0.054** | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |

Blocks 1–5 are essentially exact; **block 0's gradient is nearly orthogonal to the truth.**
The cause is catastrophic cancellation, not rounding: `x_0` is embedding-scale, but it is
recovered by subtracting six blocks' worth of accumulated residual. Running the same check in
**fp32 only improves it to 0.24** — precision alone does not fix an ill-conditioned subtraction.

This is cheap to fix: `x_0` is already an *input* to the reversible Function, so it can be stored
(one extra activation) instead of reconstructed. Implemented as `rev_mid_seed`:

| | block-0 cosine | max grad rel err | val @10M | peak alloc (b16) | max batch |
|---|---|---|---|---|---|
| `rev_mid` | 0.054 | 1.20 | **4.6716** | 1157 MB | 192 |
| `rev_mid_seed` | **0.9999** | **0.031** | 4.6731 | 1176 MB | 176 |

**The surprise: fixing it changes nothing.** Restoring block-0's gradient from "random" to exact
moves val loss by 0.0015 — noise — while costing 179 MB at batch 192, enough to drop the max
batch below 192. At 6 layers the first block is apparently doing little enough work that a
garbage gradient there is harmless. I would expect this to matter more with depth.

### 5c. Windows-specific memory traps

- **No OOM at the ceiling.** Past ~5120 MB reserved the driver falls back to shared system
  memory instead of raising. Batch 256 "succeeded" at 6382 MB reserved while throughput dropped
  ~20%. Check `max_memory_reserved` against `torch.cuda.mem_get_info()` rather than trusting the
  absence of an exception.
- **`expandable_segments:True` is a no-op on Windows.** It is the usual remedy for the ~1.2x gap
  between allocated and reserved; here it changed reserved by exactly 0 MB at every batch. That
  fragmentation gap is what actually caps the batch at 192 (allocated is only 4502 MB).

### 5d. Tuning the baseline fairly mattered more than the architecture

The first baseline run used LR 3e-3 and reached val 4.150. Re-tuned to 7.5e-4 it reaches
**3.783** — a 0.37-nat improvement, **26x larger than the entire baseline-vs-reversible gap**.
Any architecture comparison run at a single shared LR would have been meaningless here.

---

## 6. Reproducing

```bash
python data_prep.py                              # tokenizer + train.bin / val.bin
python gradcheck.py                              # fp32/CPU exactness of both stacks + fused CE
python revcheck.py --arch rev_mid --steps 300    # GPU/bf16 drift with trained weights
python train.py --arch baseline --batch 16  --lr 7.5e-4 --tokens 50e6 --out runs/r1b
python train.py --arch rev_mid  --batch 16  --lr 7.5e-4 --tokens 50e6 --out runs/r2
python train.py --arch rev_mid  --batch 192 --lr 1.3e-3 --tokens 50e6 --out runs/r3
python speed_ab.py --batch 16                    # thermally matched throughput
python thermal_bench.py                          # steady-state throughput + clock telemetry
python report.py && python plot_curves.py
```

`RESULTS.md` holds auto-generated tables for every run, including the superseded pre-fix runs
(kept under `runs/buggy_*` for the record).
