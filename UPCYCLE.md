# Dense → MoE sparse upcycling

A dense 23M GPT is trained for 20M tokens, converted into an 8-expert top-2 MoE, and
trained for another 30M tokens. A dense control continues from the same checkpoint with
the same batches and the same learning-rate schedule.

![curves](curves_upcycle.png)

## Setup

- **Data:** FineWeb-Edu with an 8k BPE vocabulary (`data/train.bin`). Batch 16 × seq 512;
  peak lr 7.5e-4, the best dense setting from `RESULTS.md`.
- **Schedule:** a single warmup + cosine schedule covers all 50M tokens. The dense phase
  runs steps 0–2441 (20M tokens). Both continuation arms run steps 2441–6103 (30M tokens).
- **Conversion** (`moe.py`):
  - Each block's MLP is copied into 8 identical experts.
  - A router is added with random init (std 0.02). Each token goes to its top 2 experts,
    and the 2 gate weights are renormalised to sum to 1.
  - Because the experts are identical and the gates sum to 1, the MoE computes exactly
    what the dense model computes at conversion.
  - The dense MLP's AdamW moments are copied into every expert; the router starts with
    empty optimizer state.
  - A Switch-style load-balancing loss (coefficient 0.01) is added to the training loss.
- **Size:** 111.2M total parameters, 35.7M active per token (the dense model is 23.1M).

## Results

| arm | tokens | val loss at 20M | final val loss (2M tokens) | ppl | tok/s |
|---|---|---|---|---|---|
| dense, phase 1 | 0 → 20M | — | 4.1593 | 64.0 | 21k* |
| dense control | 20M → 50M | 4.15564 | **3.7800** | 43.8 | 48k |
| MoE 8x top-2 (upcycled) | 20M → 50M | 4.15563 | **3.7345** | 41.9 | 25k* |

\* Speeds vary because the laptop GPU (RTX 3060, 6 GB) throttles thermally.

- **Lossless conversion:** at the conversion point the MoE's val loss is 4.15563 against
  4.15564 for the dense model (a difference of 1.5e-5, which is bf16 noise). There is no
  loss spike.
- **Training continues:** after conversion the MoE's val loss keeps falling, from 4.156
  to 3.735.
- **Better than the control by 0.045 nats:** the MoE finishes 0.045 nats below the dense
  control (perplexity 41.9 vs 43.8). The two arms are level up to about 25M tokens; after
  that the gap widens steadily and is still widening at 50M.
- **Sanity check:** the dense control ends at 3.780, which matches the earlier standalone
  dense run `r1b` (3.783), so the checkpoint/resume path is faithful.
- **Experts specialise:** each expert's distance from the mean of its layer's experts is
  0.37–0.43 of that mean's size after training. At conversion it was 0.
- **Load balance:** layers 1–5 are reasonably balanced, with experts taking 6–26% of
  routed slots against an ideal of 12.5%. Layer 0 is not: one expert takes 50% of slots
  and two experts get under 1%.

## Reproduction on Colab (Tesla T4, fp16 + GradScaler)

The same pipeline was run on a Colab T4 with
`notebooks/04_colab_dense_to_moe_upcycling.ipynb`; results are in `runs/colab_t4/`.

| | local RTX 3060, bf16 | Colab T4, fp16 |
|---|---|---|
| dense val loss at 20M tokens | 4.15564 | 4.15651 |
| MoE val loss right after conversion (difference) | 4.15563 (1.5e-5) | 4.15652 (1.3e-5) |
| dense control, final val loss | 3.7800 | 3.7798 |
| MoE, final val loss | 3.7345 | 3.7354 |
| MoE gain over dense | −0.045 | −0.044 |

The two runs agree to within 0.001, which also exercises the fp16 path. Layer 0 is
unbalanced on the T4 too, with one expert taking 50% of routed slots, so the imbalance is
systematic rather than seed noise.

## Stronger load balancing

With a coefficient of 0.01, layer 0 ends badly unbalanced: one expert takes 50% of routed
slots and two take under 1%, where an even split is 12.5%. The MoE continuation was
re-run from the same dense checkpoint with coefficients 0.05 and 0.1
(`upcycle_aux_sweep.sh`, `runs/up_moe8x2_aux*`).

![aux sweep](curves_aux_sweep.png)

| balancing coefficient | final val loss (2M tokens) | vs dense | layer-0 busiest expert | busiest expert, any layer | least-used expert |
|---|---|---|---|---|---|
| 0.01 | 3.7345 | −0.0455 | 0.500 | 0.500 | 0.004 |
| 0.05 | 3.7366 | −0.0434 | 0.289 | 0.289 | 0.037 |
| 0.1 | 3.7355 | −0.0445 | 0.236 | 0.236 | 0.049 |

- **Balance improves a lot.** At 0.1, the busiest expert in layer 0 drops from 50% to 24%
  of routed slots, and no expert in any layer falls below 4.9%. All experts are now in
  use.
- **Quality is unchanged.** The three final losses lie within 0.002 of each other, which
  is below run-to-run noise: the local and T4 runs of the same setup differed by 0.001.
- **The gain arrives later.** A stronger balancing loss holds the MoE behind the dense
  control for longer after conversion: it drops below dense at about 26–28M tokens
  instead of about 25M. It catches up by about 45M tokens.
- **Recommendation: use 0.1 as the default.** It gives the same final loss with every
  expert in use, which matters for expert-parallel throughput and for any later pruning
  or merging of experts.

## Caveats

- **Not compute-matched.** Top-2 routing doubles the MLP compute per token, and the MoE
  has 4.8× the stored parameters. The dense control shows that the gain comes from
  upcycling rather than from extra training time. It does not show that the gain beats
  a dense model of equal compute.
- **One seed per arm.**
- **Slow MoE step.** The MoE runs at about half the dense speed even on a cool GPU,
  because experts are dispatched one at a time in a Python loop rather than in a fused
  kernel.

## Reproduce

```
sh upcycle_runs.sh        # or notebooks/04_colab_dense_to_moe_upcycling.ipynb on Colab
python plot_upcycle.py
```
