# Exercise 2 — Quantize a model and measure the trade

**Goal:** shrink and speed up a model with quantization, honestly measured.

## Tasks

1. Take a trained model (your Week-5 transfer model, or a pretrained one). Record its size, latency, and held-out accuracy in float32.
2. Apply post-training quantization to INT8 (dynamic and/or static with a calibration set).
3. Re-measure size, latency, and accuracy on the *quantized* model. Report the deltas: how much smaller/faster, and how much accuracy lost?
4. If the accuracy drop is unacceptable, note that quantization-aware training would be the next step, and explain why it helps.

## Deliverable

A notebook with a before/after table (size, latency, accuracy) for float vs. INT8, and a one-line verdict on whether the trade is acceptable for a stated use case. Measuring the *actual* drop is the whole lesson.
