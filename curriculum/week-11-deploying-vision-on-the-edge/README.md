# Week 11 — Deploying vision on the edge

> **Goal:** by Sunday you can take a trained vision model and make it deployable on constrained hardware — choosing an efficient architecture, applying quantization and pruning, exporting to a portable format, and benchmarking latency, memory, and accuracy honestly.

A model that hits 95% accuracy in a notebook but needs a data-center GPU and 2 seconds per image is useless on a phone, a drone, or a $30 camera. **Edge deployment** — running vision in real time on constrained hardware — is where most real-world vision actually lives. This week you learn the efficiency toolkit: **lightweight architectures** (MobileNet, EfficientNet), **quantization** and **pruning** to shrink models, **export** to portable runtimes (ONNX, TFLite), and honest **benchmarking** of the accuracy–latency–size trade-off. Pairs naturally with [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/).

## Learning objectives

By the end of this week, you will be able to:

- **Explain** the constraints of edge hardware — compute, memory, power, latency — and how they shape model choice.
- **Choose** an efficient architecture and understand the design tricks (depthwise-separable convs) that make it small.
- **Apply** quantization (and pruning/distillation) to shrink and speed up a model.
- **Export** a model to a portable format (ONNX/TorchScript/TFLite) and run it outside PyTorch.
- **Benchmark** latency, memory, and accuracy honestly on the target, and defend a deployment choice.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Edge constraints & efficient architectures](lecture-notes/01-edge-constraints-and-architectures.md)
2. [Lecture 2 — Quantization, pruning & distillation](lecture-notes/02-quantization-pruning-distillation.md)
3. [Lecture 3 — Export, runtimes & honest benchmarking](lecture-notes/03-export-and-benchmark.md)

**Exercises**

1. [Exercise 1 — Profile an efficient vs. a heavy model](exercises/exercise-01-efficient-architecture.md)
2. [Exercise 2 — Quantize a model and measure the trade](exercises/exercise-02-quantize.md)
3. [Exercise 3 — Export to ONNX and verify parity](exercises/exercise-03-export-and-verify.md)

**Challenges**

1. [Challenge 1 — Hit a hard latency budget](challenges/challenge-01-hit-a-latency-budget.md)
2. [Challenge 2 — Make and defend a deployment decision](challenges/challenge-02-deployment-decision.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 12.

## Deliverable

A notebook/report that takes a trained model, applies quantization (and optionally pruning/distillation), exports it to a portable format, and benchmarks the accuracy–latency–size trade-off before and after — with a recommendation for a stated edge scenario.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
