# Week 11 — Quiz

Fifteen questions spanning the roofline, efficient-architecture arithmetic, the affine-quantization algebra, QAT and the straight-through estimator, distillation, honest benchmarking, and edge accelerators/TinyML. Attempt each before the answer key.

**1. In the roofline model, an operation with arithmetic intensity I below the ridge point I* = P/B is:**

- A. compute-bound, so cutting its FLOPs directly reduces latency
- B. memory-bandwidth-bound, so cutting its FLOPs alone will not reduce its latency
- C. always the fastest layer in the network
- D. unaffected by hardware and runs at peak FLOP/s

<details>
<summary>Answer</summary>

**B. memory-bandwidth-bound, so cutting its FLOPs alone will not reduce its latency** — Below the ridge, attainable performance is B·I — set by bandwidth — so latency is limited by bytes moved, not FLOPs; you must cut memory traffic (e.g. quantize) to help it.

</details>

**2. For a k x k standard convolution mapping C_in to C_out channels, the FLOP ratio of depthwise-separable to standard is:**

- A. 1/C_out + 1/k²
- B. C_out / C_in
- C. 1/k
- D. k² · C_in · C_out

<details>
<summary>Answer</summary>

**A. 1/C_out + 1/k²** — FLOPs_ds/FLOPs_std = (k²C_in + C_in C_out)/(k²C_in C_out) = 1/C_out + 1/k²; for k=3 and large C_out this is ~1/9.

</details>

**3. The energy cost of an edge inference is often dominated by:**

- A. off-chip DRAM data movement, not the arithmetic (MACs)
- B. the Python interpreter overhead
- C. the number of multiply-accumulate operations alone
- D. the size of the source code

<details>
<summary>Answer</summary>

**A. off-chip DRAM data movement, not the arithmetic (MACs)** — Moving a word from DRAM costs orders of magnitude more energy than a MAC (Horowitz, 2014), which is why reducing memory traffic — quantization, fusion — saves battery even at constant FLOPs.

</details>

**4. EfficientNet's compound scaling scales depth, width, and resolution under the constraint:**

- A. alpha · beta² · gamma² ~ 2, so each unit of the coefficient roughly doubles FLOPs
- B. the learning rate is scaled with model size
- C. only depth is scaled; width and resolution are fixed
- D. alpha + beta + gamma = 1

<details>
<summary>Answer</summary>

**A. alpha · beta² · gamma² ~ 2, so each unit of the coefficient roughly doubles FLOPs** — Width and resolution enter FLOPs quadratically, so Tan & Le (2019) constrain alpha·beta²·gamma²~2 to make each phi-step double compute while balancing all three dimensions.

</details>

**5. In affine quantization q = clip(round(x/s) + z), the zero-point z exists primarily to:**

- A. convert the model to FP16
- B. double the model size
- C. represent real zero exactly with an integer (so zero-padding stays exact)
- D. remove the need for a scale factor

<details>
<summary>Answer</summary>

**B. double the model size** — z is the integer encoding of real 0; getting zero exact matters for padding and ReLU. Symmetric quantization sets z=0 and is used for weights.

</details>

**6. Quantization error decomposes into two components that trade off via the range choice:**

- A. bias and variance of the labels
- B. latency and throughput
- C. rounding error (~s/2) and clipping error (values outside the range)
- D. training error and test error

<details>
<summary>Answer</summary>

**C. rounding error (~s/2) and clipping error (values outside the range)** — A wider range lowers clipping but raises the bin width s and hence rounding noise (variance s²/12); calibration is choosing that range.

</details>

**7. Per-channel (vs. per-tensor) weight quantization helps mainly because:**

- A. it removes the need for calibration
- B. it converts weights to floating point
- C. it makes the model larger and thus more accurate
- D. each output filter gets its own scale, so one outlier channel no longer forces a coarse scale on all

<details>
<summary>Answer</summary>

**D. each output filter gets its own scale, so one outlier channel no longer forces a coarse scale on all** — Depthwise/conv weight distributions vary widely across channels; per-channel scales recover most of the accuracy at negligible cost, so it is standard for convolutions.

</details>

**8. Quantization-aware training uses the straight-through estimator to:**

- A. avoid quantizing anything during the forward pass
- B. make the model FP32 at inference
- C. pass gradients through the non-differentiable rounding by treating round' = 1 inside the clip range
- D. eliminate the need for any training data

<details>
<summary>Answer</summary>

**C. pass gradients through the non-differentiable rounding by treating round' = 1 inside the clip range** — The quantizer's true derivative is zero a.e.; the STE (Bengio et al., 2013) approximates it as identity inside the representable range so gradients flow and weights pre-compensate for rounding.

</details>

**9. Structured pruning is preferred over unstructured pruning on edge hardware because:**

- A. it never requires fine-tuning afterward
- B. it increases accuracy
- C. it always compresses more per parameter
- D. removing whole filters/channels genuinely shrinks the model, giving real speedups on dense hardware

<details>
<summary>Answer</summary>

**D. removing whole filters/channels genuinely shrinks the model, giving real speedups on dense hardware** — Unstructured sparsity leaves the dense tensor shape unchanged, so standard kernels get no speedup; structured pruning shrinks the actual dimensions.

</details>

**10. In knowledge distillation, the soft-target loss is scaled by T² because:**

- A. softening with temperature T shrinks the gradient magnitude by ~1/T², so T² restores comparable scale
- B. the teacher has twice as many layers
- C. it doubles the learning rate
- D. it converts KL divergence into cross-entropy

<details>
<summary>Answer</summary>

**A. softening with temperature T shrinks the gradient magnitude by ~1/T², so T² restores comparable scale** — Hinton et al. (2015) scale the distillation term by T² so its gradients stay comparable to the hard-label cross-entropy term when both are combined.

</details>

**11. The single most common *silent* deployment failure is:**

- A. preprocessing mismatch between training and the deployed pipeline
- B. a model file that is slightly too large
- C. having too few layers
- D. using ONNX as the export format

<details>
<summary>Answer</summary>

**A. preprocessing mismatch between training and the deployed pipeline** — Different resize, channel order (BGR/RGB), or normalization collapses accuracy in production while offline tests on correctly-preprocessed data still pass; verify preprocessing parity end-to-end.

</details>

**12. Operator fusion (e.g. folding conv+BN+ReLU into one kernel) speeds inference mainly by:**

- A. reducing memory traffic — intermediate activations never round-trip to DRAM — a memory-bound win
- B. adding more parameters
- C. increasing the number of FLOPs
- D. converting the model to a larger dtype

<details>
<summary>Answer</summary>

**A. reducing memory traffic — intermediate activations never round-trip to DRAM — a memory-bound win** — Fusion raises arithmetic intensity by cutting bytes moved; since edge inference is often memory-bound, it is frequently the biggest wall-clock win.

</details>

**13. Benchmarking latency honestly for a real-time edge stream requires reporting:**

- A. median and tail (p95/p99) after warmup, on the target, for the shipped artifact
- B. the FP32 accuracy of the pre-export model
- C. only the mean latency
- D. a single timed call on the dev GPU

<details>
<summary>Answer</summary>

**A. median and tail (p95/p99) after warmup, on the target, for the shipped artifact** — A 30 fps stream needs the p99 under 33 ms; warmup removes one-time costs, and the shipped (exported, quantized) model on the target is the only meaningful thing to measure.

</details>

**14. On an INT8-first NPU (e.g. Edge TPU, Neural Engine), an unsupported operator typically causes:**

- A. automatic conversion of the whole model to FP64
- B. the model to run twice as fast
- C. a fallback to the CPU, which — if the graph ping-pongs — can be slower than staying on CPU
- D. the accelerator to increase its clock speed

<details>
<summary>Answer</summary>

**C. a fallback to the CPU, which — if the graph ping-pongs — can be slower than staying on CPU** — Accelerators only help if the whole graph stays on their happy path; fallbacks add transfer overhead, so keeping ops supported (and quantized) is core deployment engineering.

</details>

**15. For a vision model targeting a microcontroller, the binding constraint is usually:**

- A. the number of FLOPs
- B. the on-disk size of the training script
- C. peak activation (SRAM) memory, which a single high-resolution early feature map can exceed
- D. the parameter count alone

<details>
<summary>Answer</summary>

**A. the number of FLOPs** — MCUs have only hundreds of KB of SRAM; MCUNet (Lin et al., 2020) shows peak activation memory — not FLOPs or weights — is what must be engineered down, via resolution cuts and patch-based execution.

</details>

---
