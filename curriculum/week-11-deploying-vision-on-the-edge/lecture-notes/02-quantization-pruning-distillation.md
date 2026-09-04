# Lecture 2 — Quantization, pruning & distillation: the algebra of compression

Beyond choosing a small architecture, three techniques *compress* a trained model further — often
dramatically — at careful, measured accuracy cost. Quantization is the highest-leverage; pruning and
distillation complement it. This lecture gives the algebra, not just the recipes.

## Affine quantization from first principles

Models train in FP32, but that precision is overkill for inference. **Quantization** represents a real value
`x` in a low-bit integer. The standard **affine (asymmetric) scheme** maps a float range `[x_min, x_max]` onto
`b`-bit integers `[q_min, q_max]` (for INT8, `[-128, 127]` or `[0, 255]`) via a **scale** `s` and an integer
**zero-point** `z`:

    s = (x_max − x_min) / (q_max − q_min),   z = round(q_min − x_min / s),
    q = clip( round(x / s) + z,  q_min,  q_max),   x_hat = s · (q − z).

`s` is the width of one quantization bin; `z` is the integer that represents real zero exactly (important so
that zero-padding stays exact). **Symmetric** quantization forces `z = 0` and a range `[-a, a]`, which is
simpler and lets hardware skip the zero-point term — preferred for *weights*. **Asymmetric** captures the
one-sided range of post-ReLU *activations* better, so a common recipe is symmetric weights + asymmetric
activations (Jacob et al., 2018, CVPR, "Quantization and training of neural networks for efficient
integer-arithmetic-only inference"; Krishnamoorthi, 2018, whitepaper).

The quantization error has two sources: **rounding error**, bounded by `s/2` per value (uniform in
`[-s/2, s/2]`, so variance `s²/12`), and **clipping error** for values outside `[x_min, x_max]`. These
trade off: a wider range lowers clipping but raises `s`, hence rounding noise. Choosing the range well is the
whole game of calibration.

## PTQ vs. QAT, and calibration

- **Post-training quantization (PTQ).** Quantize an already-trained model; no retraining. Fast and easy,
  loses a little accuracy. It needs a small **calibration** set (a few hundred unlabeled images) to estimate
  activation ranges. Calibration methods: **min-max** (use observed extremes; sensitive to outliers),
  **percentile** (clip the top 0.1% to trade clipping for rounding), and **entropy/KL** (choose the clip that
  minimizes the KL divergence between the FP32 and quantized activation histograms — TensorRT's default).
- **Quantization-aware training (QAT).** Simulate quantization *during* fine-tuning so the network learns
  weights robust to it — covered in depth in Lecture 4. Use it when PTQ's drop is unacceptable.

**Per-tensor vs. per-channel.** One `(s, z)` per tensor is cheapest but a single outlier channel forces a
coarse `s` on all channels. **Per-channel** weight quantization gives each output filter its own scale,
recovering most of the accuracy at negligible cost — standard for convolutions.

```python
import torch
# Dynamic PTQ (activations quantized on the fly; simplest, good for Linear-heavy nets):
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

INT8 gives ~**4x** smaller weights and, on hardware with INT8 MAC units, real speed and energy wins (integer
multiply-accumulate is cheaper, and half the bytes move). Even INT4/binary exist for extreme budgets, at more
accuracy cost. Always **measure** the drop — quantization is a trade, not a free lunch.

## Pruning: remove what does not matter

Trained networks are over-parameterized, so many weights can go. **Magnitude pruning** removes the
smallest-magnitude weights (Han et al., 2015, "Learning both weights and connections"). Two regimes:

- **Unstructured pruning** zeros individual weights → high sparsity, but standard dense hardware ignores
  the zeros, so you get *no speedup* without sparse-kernel support. The **Lottery Ticket Hypothesis**
  (Frankle & Carbin, 2019, ICLR) shows such sparse subnetworks *can* train to full accuracy from the original
  init — a striking scientific result, but mostly about trainability, not edge speed.
- **Structured pruning** removes whole filters/channels → the model genuinely shrinks, giving **real
  speedups on any hardware**. Less compression per parameter, but it is the practical edge choice.

Prune, then **fine-tune** to recover accuracy, iterating gradually rather than in one aggressive cut.

## Distillation: train a small student to mimic a big teacher

**Knowledge distillation** (Hinton, Vinyals & Dean, 2015, "Distilling the knowledge in a neural network")
trains a small **student** to reproduce a large **teacher's** *soft* output distribution, not just hard
labels. Soften both with a **temperature** `T`: `p_i = softmax(z_i / T)`. The loss combines a distillation
term (KL to the teacher's softened outputs) and a standard cross-entropy on true labels:

    L = (1 − alpha) · CE(y, student) + alpha · T² · KL( teacher_T ‖ student_T ).

The `T²` factor rescales the gradient magnitude, which shrinks with `1/T²`, to keep it comparable to the CE
term. The soft targets carry **dark knowledge** — the teacher's relative confidence across wrong classes
(that a "3" looks a bit like an "8") — which is a richer supervisory signal than a one-hot label, so the
student often beats the same architecture trained on labels alone. DeiT (Touvron et al., 2021) distilled ViTs
into efficient students exactly this way (Week 10).

## Stacking them, in order

These compose: pick an efficient architecture → optionally distill from a bigger teacher → structured prune →
quantize. Each step trades a little accuracy for size/speed, and you **measure at every step**. Order matters:
prune and distill on the FP32 model, quantize last, and re-benchmark the shipped artifact.

```mermaid
flowchart LR
  A["Efficient architecture"] --> B["Optional: distill from teacher"]
  B --> C["Structured pruning + fine-tune"]
  C --> D["Quantize to INT8 (PTQ or QAT)"]
  D --> E["Measure accuracy at every step"]
```
*The compression pipeline stacks, and each stage is a measured accuracy-for-efficiency trade.*

**Takeaway:** compression rests on three measured trades. Affine quantization `q = clip(round(x/s)+z)` maps a
float range to integers; its error splits into rounding (`~s/2`, variance `s²/12`) and clipping, so calibration
(min-max, percentile, or KL) is choosing that range, and per-channel symmetric weights + asymmetric activations
is the standard recipe (Jacob et al., 2018). Structured pruning gives real edge speedups where unstructured does
not; distillation transfers a teacher's dark knowledge via temperature-`T` soft targets with a `T²`-scaled KL
loss (Hinton et al., 2015). They stack — measure the accuracy drop at every step.
