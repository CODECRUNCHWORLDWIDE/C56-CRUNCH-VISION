# Exercise 4 — Quantization-aware training with the straight-through estimator

**Goal:** internalize Lecture 4 — recover PTQ's accuracy loss by training *through* the quantizer, and
see the STE at work.

## Part A — from scratch (understand the STE)

1. Implement a **fake-quant** function `fq(x, s, z)` that in the forward pass computes
   `s*(clip(round(x/s)+z, qmin, qmax) - z)` and in the backward pass passes the gradient straight through for
   inputs inside the clip range and zeros it outside (use a custom `torch.autograd.Function`). Verify with
   `gradcheck`-style probes that the backward is the STE, not the true (zero) derivative.
2. On a tiny 1-layer regression or classification, confirm that a network with your fake-quant node can still
   be trained — gradients flow — whereas replacing STE with the true rounding derivative stalls learning.

## Part B — QAT on a real model

1. Take a model where PTQ dropped accuracy noticeably (from Exercise 2). Insert fake-quant observers (or use
   the framework's QAT API), fine-tune for a few epochs, then convert to a real INT8 model.
2. Compare final INT8 accuracy for **PTQ vs. QAT** at equal bit-width, and report the recovered gap. Note the
   training cost QAT added.
3. Bonus: fold Conv+BN before quantizing and confirm training-time and inference-time numerics agree — or
   show the accuracy bug that appears if you forget to fold.

## Deliverable

A notebook with (i) your from-scratch STE fake-quant and the proof it enables gradient flow, and (ii) a
PTQ-vs-QAT accuracy comparison on a real model showing QAT recovering the drop, with one paragraph explaining
why the biased STE gradient nonetheless trains a better-quantizing network.
