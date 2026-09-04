# Exercise 2 — Quantize, calibrate, and measure the trade honestly

**Goal:** shrink and speed up a model with quantization, and see how calibration and granularity change
the accuracy cost.

## Tasks

1. Take a trained model (your Week-5 transfer model, or a pretrained one). Record its FP32 size, latency
   (median + p95), and held-out accuracy as the baseline.
2. Apply **dynamic** PTQ (INT8) and re-measure size, latency, accuracy. Then apply **static** PTQ with a
   calibration set: try **min-max** and **percentile/entropy** calibration and compare their accuracy.
3. Try **per-tensor vs. per-channel** weight quantization and report the accuracy difference — per-channel
   should recover most of the drop at negligible cost.
4. Report a before/after table (size, latency, accuracy) for each configuration and compute the deltas: how
   much smaller/faster, and how much accuracy lost. If any drop is unacceptable, state precisely why QAT
   (Lecture 4) would help.

## Deliverable

A notebook with a multi-row table (FP32, dynamic INT8, static min-max, static entropy, per-channel), a one-line
verdict on the best trade for a stated use case, and a sentence explaining the rounding-vs-clipping reason one
calibration beat another. Measuring the *actual* drop is the whole lesson.
