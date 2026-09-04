# Week 11 — Deploying vision on the edge

> **Goal:** by Sunday you can (1) reason about edge inference as a roofline-constrained optimization — reading whether a layer is compute- or memory-bound and predicting latency from FLOPs, bytes, and arithmetic intensity rather than guessing; (2) derive affine INT8 quantization from first principles (scale, zero-point, clipping error), choose PTQ calibration honestly, and train through quantization with the straight-through estimator when PTQ drops too far; (3) apply structured pruning and knowledge distillation as *measured* accuracy-for-efficiency trades; and (4) export to a portable runtime, prove output and preprocessing parity, and benchmark the shipped artifact on the target with median-and-tail latency, peak memory, and held-out accuracy — then defend a deployment pick against a hard constraint.

A model that hits 95% accuracy in a notebook but needs a data-center GPU and two seconds per image is useless on a phone, a drone, or a $30 smart camera. **Edge deployment** — running vision in real time on constrained hardware — is where most real-world computer vision actually lives, and it is a different discipline from training. The objective flips: instead of maximizing accuracy with compute to burn, you **maximize accuracy subject to hard latency, memory, and power ceilings**. That is a constrained optimization, and this week gives you both the toolkit and the theory to solve it.

The undergraduate version of this week says 'use MobileNet, quantize to INT8, export to ONNX, measure the trade.' We will not stop there. You will derive *why* depthwise-separable convolution cuts a 3x3 layer's cost roughly 8-9x, and read the **roofline model** (Williams, Waterman & Patterson, 2009) to know whether that saving even helps on your target or whether you are memory-bandwidth-bound. You will derive the **affine quantization** map q = round(x/s) + z from the requirement of representing a float range in 256 integer levels, see where clipping and rounding error come from, and understand the **straight-through estimator** (Bengio et al., 2013) that makes quantization-aware training differentiable through a non-differentiable rounding step. You will treat pruning and distillation as measured trades, not folklore, and you will learn to benchmark like a scientist: warmup, median *and* tail latency, the shipped (exported, quantized) artifact, on the real target.

Above all you will internalize the field's most expensive silent bug — **preprocessing mismatch** between training and the deployed pipeline — and the engineering honesty that separates a system that works in the field from one that only worked in the notebook. This is the last piece before the Week 12 capstone. Pairs naturally with [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) for serving and production monitoring.

## Learning objectives

By the end of this week, you will be able to:

- **Analyze** edge inference with the roofline model — computing arithmetic intensity, classifying each layer as compute- or memory-bound, and predicting whether a FLOP reduction will actually reduce latency on a given target.
- **Derive** the FLOP and parameter cost of standard vs. depthwise-separable convolution, and explain MobileNet's inverted residuals and EfficientNet's compound-scaling constraint from the arithmetic.
- **Formulate** affine INT8 quantization (scale, zero-point, clipping), decompose its error into rounding and clipping components, and choose between symmetric/asymmetric and per-tensor/per-channel schemes with reasons.
- **Implement** quantization-aware training with fake-quant nodes and the straight-through estimator, and state precisely why it recovers accuracy that post-training quantization loses.
- **Apply** structured pruning and knowledge distillation as measured trades — deriving the distillation loss with temperature and explaining why soft targets carry more information than hard labels.
- **Export** a model to a portable runtime (ONNX/TorchScript/TFLite), verify numerical output parity and end-to-end preprocessing parity, and diagnose an unsupported-operator or normalization mismatch.
- **Benchmark** the shipped artifact honestly on the target — warmup, median and p95/p99 latency, peak memory, on-disk size, and held-out accuracy — and report the accuracy-latency-size frontier.
- **Defend** a deployment configuration against a stated hard constraint with a decision matrix, and name the production risks (distribution shift, monitoring) you would guard before shipping.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | Past the outcome set: nothing in this ledger entry asks for INT8 quantization, a portable export, or a latency budget measured on target hardware. This week is added, not mapped. |
| Industry | Ship a model inside a stated latency, memory and size budget, and prove the exported artifact still produces the same numbers as the one that was trained. |
| Beyond the bar | implements quantization-aware training with fake-quant nodes and the straight-through estimator rather than accepting what post-training quantization loses — `exercises/exercise-04-qat-straight-through.md` |

## Prerequisites

- The previous week's mini-project (Vision Transformers), committed and working.
- A trained vision model you can reuse — ideally your Week-5 transfer-learning model.
- Comfort with convolution arithmetic (kernel size, channels, stride) and basic PyTorch (`nn.Module`, `state_dict`, inference).
- Enough numerics to reason about float vs. integer representation and rounding.

## This week

**Lectures**

1. [Lecture 1 — Edge constraints, the roofline, & efficient architectures](lecture-notes/01-edge-constraints-and-architectures.md)
2. [Lecture 2 — Quantization, pruning & distillation: the algebra of compression](lecture-notes/02-quantization-pruning-distillation.md)
3. [Lecture 3 — Export, runtimes & statistically honest benchmarking](lecture-notes/03-export-and-benchmark.md)
4. [Lecture 4 — Quantization theory, QAT with the straight-through estimator, & hardware-aware co-design](lecture-notes/04-quantization-theory-and-co-design.md)
5. [Lecture 5 — Compilers, runtimes, edge accelerators, & TinyML](lecture-notes/05-compilers-accelerators-tinyml.md)

**Exercises**

1. [Exercise 1 — Profile efficient vs. heavy models against the roofline](exercises/exercise-01-efficient-architecture.md)
2. [Exercise 2 — Quantize, calibrate, and measure the trade honestly](exercises/exercise-02-quantize.md)
3. [Exercise 3 — Export to ONNX, verify output parity, and break preprocessing on purpose](exercises/exercise-03-export-and-verify.md)
4. [Exercise 4 — Quantization-aware training with the straight-through estimator](exercises/exercise-04-qat-straight-through.md)

**Challenges**

1. [Challenge 1 — Hit a hard latency budget (constrained optimization)](challenges/challenge-01-hit-a-latency-budget.md)
2. [Challenge 2 — Make and defend a deployment decision with a decision matrix](challenges/challenge-02-deployment-decision.md)
3. [Challenge 3 — TinyML memory wall or compiler autotuning (open, research-flavored)](challenges/challenge-03-tinyml-or-compiler-open.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 12.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A notebook/report that takes a trained vision model, records an honest float32 baseline, applies at least INT8 quantization (optionally an efficient architecture, structured pruning, or distillation) measuring the accuracy/size/latency delta at each step, exports to a portable runtime with verified output *and* preprocessing parity (including a deliberate-mismatch failure demo), benchmarks the shipped artifact on the target with median-and-tail latency, and ends in a justified deployment recommendation for a stated edge scenario with a hard constraint.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
