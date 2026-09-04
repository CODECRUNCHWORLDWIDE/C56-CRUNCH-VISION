# Week 11 — Homework

Cement the deployment toolkit — efficiency arithmetic, the quantization algebra, QAT, and honest benchmarking — before the capstone. Do the derivations by hand before coding: the FLOP and quantization math must be instinct before a library hides it, and the benchmarking discipline is what makes your capstone numbers trustworthy.

## Tasks

- Derive, on paper, the FLOP and parameter cost of standard vs. depthwise-separable convolution, obtain the ratio 1/C_out + 1/k², and explain — using arithmetic intensity — why the measured speedup is usually smaller than this FLOP ratio.
- Work the affine-quantization algebra: for a stated float range, compute the scale and zero-point for symmetric and asymmetric INT8, quantize/dequantize a value, and state the rounding vs. clipping error trade that calibration resolves.
- Explain post-training quantization vs. quantization-aware training: what the straight-through estimator does in QAT's forward and backward passes, and precisely when the extra training cost is worth it.
- Write a preprocessing-parity checklist (resize interpolation, crop, channel order, normalization mean/std, scaling) for verifying a deployed pipeline matches training, and describe the deliberate-mismatch test you would add.
- Design an honest benchmarking protocol for a real-time edge target: warmup policy, batch size, median and tail statistics, what artifact to measure, and what environment to control — and justify each choice.
- Read the MobileNet, Jacob et al. quantization, and MCUNet references (in the reading list) and write one paragraph on which technique you will apply in your Week-12 capstone and why, given a specific edge target.

## Definition of done

A committed project that: records an honest FP32 baseline (size, median+tail latency, memory, accuracy) on a stated target; applies at least INT8 quantization (with calibration/granularity choices justified, and QAT if PTQ dropped too far) measuring the delta at each step; exports to a portable runtime with verified output *and* preprocessing parity including a deliberate-mismatch demo; benchmarks the shipped, quantized artifact on the target and plots the accuracy-latency-size frontier; and ends in a defended deployment recommendation for a stated hard-constraint scenario with named production risks.

Submit by committing your work to your course repo under `week-11/`.
