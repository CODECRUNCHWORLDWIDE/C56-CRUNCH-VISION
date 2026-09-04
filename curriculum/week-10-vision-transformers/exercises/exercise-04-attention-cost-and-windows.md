# Exercise 4 — Measure the N^2 cost and implement windowed attention

**Goal:** make the quadratic-cost problem and the Swin fix concrete and measurable, not abstract.

## Tasks

1. Implement scaled dot-product attention `softmax(QK^T/sqrt(d))V` from the definition (no
   `nn.MultiheadAttention`). Verify against `torch.nn.functional.scaled_dot_product_attention` with
   `allclose`.
2. **The sqrt(d) ablation.** Feed random Q, K with `d in {8, 64, 512}` and, with and without the `1/sqrt(d)`
   scale, print the standard deviation of the logits and the max softmax weight. Show that without the scale
   the logit std grows like `sqrt(d)` and softmax saturates toward one-hot — the Lecture 2 variance argument.
3. **Cost curve.** For `N in {49, 196, 784, 3136}` (patch sizes 32, 16, 8, 4 at 224px), time a forward pass
   and record peak memory. Plot cost vs. `N` on log-log axes and fit the slope; confirm it matches the
   `N^2` prediction (slope ~2 for the attention term).
4. **Windowed attention.** Partition the `N` tokens into non-overlapping `M x M` windows (say `M = 7`),
   apply attention within each window only, and re-measure cost. Show it scales *linearly* in `N` for fixed
   `M`. Then implement the **shifted** partition (roll by `M/2`) for the alternate layer and explain, in a
   sentence, why the shift is needed for cross-window information flow (Swin, Lecture 4).

## Deliverable

A notebook with: an `allclose`-verified attention implementation, the sqrt(d) saturation ablation, a
log-log cost curve confirming the `N^2` slope, and a windowed-attention variant demonstrating linear cost
plus a working shift. The number 16x (halving patch size) should appear and be verified.
