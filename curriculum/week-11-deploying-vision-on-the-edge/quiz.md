# Week 11 — Quiz

Ten questions. Answer key below.

**1. The constraints that define edge deployment are:**

- A. Compute, memory, power, and latency
- B. Only accuracy
- C. Dataset size
- D. Only model size

**2. Depthwise-separable convolution reduces cost by:**

- A. Removing all convolutions
- B. Adding more channels
- C. Using bigger kernels
- D. Splitting a conv into a depthwise spatial filter plus a pointwise 1×1 channel mix

**3. EfficientNet's compound scaling scales:**

- A. Only depth
- B. Only the learning rate
- C. Depth, width, and resolution together in a principled ratio
- D. Only the batch size

**4. INT8 quantization typically makes a model:**

- A. 4× larger
- B. Perfectly lossless
- C. Slower always
- D. ~4× smaller and often faster, with some accuracy trade

**5. Quantization-aware training (QAT) differs from post-training quantization by:**

- A. Being always worse
- B. Simulating quantization during training so the model adapts, preserving more accuracy
- C. Needing no data
- D. Using 32-bit weights

**6. Structured pruning (removing whole filters/channels) is preferred on edge because:**

- A. It yields real speedups on standard hardware, unlike unstructured sparsity
- B. It increases accuracy
- C. It compresses the most
- D. It needs no fine-tuning

**7. Knowledge distillation trains:**

- A. Two identical models
- B. No model at all
- C. A small student to mimic a large teacher's soft outputs
- D. A big model on a small one

**8. ONNX is:**

- A. A framework-neutral model format for portable deployment
- B. A dataset
- C. A GPU brand
- D. A quantization method

**9. The #1 silent deployment failure is:**

- A. Using ONNX
- B. A small model
- C. Preprocessing mismatch between training and the deployed pipeline
- D. Too many layers

**10. Deployment accuracy should be measured on:**

- A. The original float model
- B. The training set
- C. The exported, quantized model (the version actually shipped) on held-out data
- D. Any model

---

## Answer key

1. **A. Compute, memory, power, and latency** — Edge hardware is limited on all four, pushing toward small, fast models.
2. **D. Splitting a conv into a depthwise spatial filter plus a pointwise 1×1 channel mix** — The factorization cuts a 3×3 conv's cost roughly 8–9×, powering MobileNet.
3. **C. Depth, width, and resolution together in a principled ratio** — Balancing all three yields the best accuracy per FLOP.
4. **D. ~4× smaller and often faster, with some accuracy trade** — 32→8 bits shrinks and speeds the model; measure the accuracy drop.
5. **B. Simulating quantization during training so the model adapts, preserving more accuracy** — QAT trades extra training effort for better quantized accuracy than PTQ.
6. **A. It yields real speedups on standard hardware, unlike unstructured sparsity** — Removing whole channels genuinely shrinks the model; unstructured sparsity often doesn't speed up.
7. **C. A small student to mimic a large teacher's soft outputs** — The student learns from the teacher's richer distributions, often beating label-only training.
8. **A. A framework-neutral model format for portable deployment** — Export to ONNX and run almost anywhere via ONNX Runtime.
9. **C. Preprocessing mismatch between training and the deployed pipeline** — Different resize/normalize/channel-order collapses accuracy while offline tests pass — verify parity.
10. **C. The exported, quantized model (the version actually shipped) on held-out data** — Benchmark the shipped artifact on the target; the pre-optimization number isn't your production accuracy.
