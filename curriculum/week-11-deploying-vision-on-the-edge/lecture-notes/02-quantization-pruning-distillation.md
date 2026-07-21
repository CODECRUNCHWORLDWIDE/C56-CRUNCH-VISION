# Lecture 2 — Quantization, pruning & distillation

Beyond choosing a small architecture, three techniques *compress* a trained model further — often dramatically — with careful, measured accuracy trade-offs. Quantization is the highest-leverage one; pruning and distillation complement it.

## Quantization: fewer bits per number

Models train in 32-bit floats, but that precision is overkill for inference. **Quantization** stores and computes weights (and often activations) in lower precision — typically **8-bit integers (INT8)**:
- **~4× smaller** model (32→8 bits).
- **Faster** — integer arithmetic is cheaper and many edge chips have dedicated INT8 accelerators.
- **Lower power.**

Two flavors:
- **Post-training quantization (PTQ)** — quantize an already-trained model. Fast and easy; may lose a little accuracy. A small *calibration* set estimates the value ranges to map float→int well.
- **Quantization-aware training (QAT)** — simulate quantization *during* training/fine-tuning so the model learns to be robust to it. More work, better accuracy — use when PTQ drops too much.

```python
import torch
# Dynamic post-training quantization (simplest form):
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

Even lower precision (INT4, binary) exists for extreme constraints, at more accuracy cost. Always **measure** the accuracy drop — quantization is a trade, not a free lunch.

## Pruning: remove what doesn't matter

**Pruning** removes weights (or whole channels/filters) that contribute little, exploiting the fact that trained networks are over-parameterized.
- **Unstructured pruning** zeros individual weights — high compression, but needs sparse-compute support to actually speed up (often it doesn't on standard hardware).
- **Structured pruning** removes entire filters/channels — less compression but *real* speedups on any hardware, since the model genuinely shrinks.
Prune, then **fine-tune** to recover accuracy, iterating. Structured pruning is usually the practical choice for edge.

## Distillation: train a small model to mimic a big one

**Knowledge distillation** trains a small **student** model to reproduce a large **teacher's** outputs (its soft probability distributions, richer than hard labels). The student learns from the teacher's 'dark knowledge' and often beats the same small model trained on labels alone. Distillation is how many deployable models are made (DeiT distilled ViTs into efficient students, Week 10).

## Combining them

These stack: pick an efficient architecture → optionally distill from a bigger teacher → prune → quantize. Each step trades a little accuracy for size/speed, and you **measure at every step** to stay on the right side of the trade. The order and aggressiveness depend on your accuracy budget.

```mermaid
flowchart LR
  A["Pick an efficient architecture"] --> B["Optional - distill from a bigger teacher"]
  B --> C["Structured pruning"]
  C --> D["Quantization to INT8"]
  D --> E["Measure accuracy at every step"]
```
*The compression pipeline stacks architecture choice, optional distillation, pruning, and quantization, with accuracy measured along the way.*

**Takeaway:** compress a trained model with three complementary tools — quantization (fewer bits, ~4× smaller and faster, via PTQ or accuracy-preserving QAT), pruning (remove low-value weights/filters; structured pruning gives real speedups), and distillation (train a small student to mimic a big teacher). They stack, and each is a measured accuracy-for-efficiency trade — never assume the drop is zero; benchmark it.
