# Exercise 1 — Profile an efficient vs. a heavy model

**Goal:** feel the architecture-level efficiency difference.

## Tasks

1. Load a heavy model (ResNet-50) and an efficient one (MobileNetV3 or EfficientNet-B0), both pretrained.
2. For each, measure: parameter count, model size (MB), and inference latency on your CPU (median over many runs, after warmup).
3. Evaluate both on a small labeled set for accuracy. Build the accuracy-vs-latency and accuracy-vs-size comparison.
4. Inspect a depthwise-separable conv block in the efficient model and explain, with the FLOP formula, why it's cheaper than a standard conv.

## Deliverable

A notebook with a table (params, size, latency, accuracy) for both models and a short explanation of depthwise-separable convolutions' savings. The efficient model should be far smaller/faster at modest accuracy cost.
