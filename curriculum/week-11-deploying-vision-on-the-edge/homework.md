# Week 11 — Homework

Cement the deployment toolkit before the capstone.

## Tasks

- Explain, with the FLOP formula, why depthwise-separable convolutions are so much cheaper than standard convs.
- Describe the difference between post-training quantization and quantization-aware training and when to use each.
- Write a checklist for verifying preprocessing parity between training and a deployed pipeline.
- Read the MobileNet and quantization docs (in resources); note one thing you'll apply in your capstone.

## Definition of done

A committed project that takes a trained vision model, applies at least quantization (optionally pruning/distillation and an efficient architecture), exports it to a portable format with verified output and preprocessing parity, and benchmarks the accuracy–latency–size trade-off before and after on the target, ending in a justified deployment recommendation for a stated scenario.

Submit by committing your work to your course repo under `week-11/`.
