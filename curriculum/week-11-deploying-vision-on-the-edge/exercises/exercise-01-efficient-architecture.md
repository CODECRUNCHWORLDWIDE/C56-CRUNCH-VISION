# Exercise 1 — Profile efficient vs. heavy models against the roofline

**Goal:** feel the architecture-level efficiency difference *and* learn why FLOPs are only a proxy for
latency.

## Tasks

1. Load a heavy model (ResNet-50) and two efficient ones (MobileNetV3-Small and EfficientNet-B0), all
   pretrained. For each, measure and tabulate: parameter count, on-disk size (MB), theoretical FLOPs (use a
   profiler such as `fvcore` or `ptflops`), and **CPU inference latency** — median and p95 over >=100 runs,
   after >=10 warmup runs, batch size 1.
2. Evaluate all three on a small labeled validation set for top-1 accuracy. Build accuracy-vs-latency and
   accuracy-vs-size scatter plots and identify the Pareto frontier.
3. **FLOPs vs. reality.** Compute each model's `latency / FLOPs`. Do the efficient models get the full FLOP
   reduction as a latency reduction? Explain any gap using Lecture 1's roofline — which layers (depthwise,
   1x1) are memory-bound and thus limited by bandwidth, not compute.
4. Inspect a depthwise-separable block in MobileNetV3 and, with the formula `FLOPs_ds/FLOPs_std = 1/C_out +
   1/k²`, compute the theoretical saving for one concrete block and compare it to the measured per-layer time
   if you can profile it.

## Deliverable

A notebook with the params/size/FLOPs/latency/accuracy table for all three models, the two Pareto plots, and a
paragraph explaining — with the roofline — why the measured speedup is smaller than the FLOP ratio predicts.
