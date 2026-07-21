# Week 10 — Quiz

Ten questions. Answer key below.

**1. A Vision Transformer turns an image into:**

- A. A single vector
- B. A sequence of patch tokens (each patch a 'word')
- C. A pixel-wise mask
- D. A bounding box

**2. Patch embedding can be implemented as:**

- A. A recurrent layer
- B. A pooling layer only
- C. A softmax
- D. A non-overlapping convolution with kernel = stride = patch size

**3. Positional embeddings are needed because self-attention is:**

- A. Too slow
- B. Permutation-invariant (has no built-in sense of order/position)
- C. Only for text
- D. Non-differentiable

**4. The [CLS] token's role is to:**

- A. Summarize the whole image for the classification head
- B. Store the learning rate
- C. Hold the positional embedding
- D. Mark the image border

**5. Self-attention lets each patch:**

- A. Attend to every other patch, giving a global receptive field in one layer
- B. Ignore all other patches
- C. Only attend to itself
- D. See only its neighbors

**6. Attention computes output as:**

- A. A running average
- B. A convolution
- C. softmax(QKᵀ/√d)V — a relevance-weighted sum of values
- D. A max pool

**7. Self-attention's cost grows as:**

- A. N² (quadratic in the number of patches)
- B. constant
- C. log N
- D. N (linear in patches)

**8. The Swin Transformer reduces cost by:**

- A. Ignoring positions
- B. Removing attention
- C. Computing attention within shifted local windows and building a hierarchy
- D. Using only one head

**9. Trained from scratch on ImageNet-1k, a ViT vs. a comparable CNN typically:**

- A. Always wins
- B. Cannot train
- C. Is identical
- D. Underperforms, because it lacks the CNN's data-efficient inductive biases

**10. A fair way to choose ViT vs. CNN for a project is to:**

- A. Pick whichever is trendier
- B. Measure held-out accuracy and cost for your data and budget (try both)
- C. Always pick ViT
- D. Always pick CNN

---

## Answer key

1. **B. A sequence of patch tokens (each patch a 'word')** — The key move: chop the image into patches and treat each as a token.
2. **D. A non-overlapping convolution with kernel = stride = patch size** — A stride-16 16×16 conv literally produces the patch embeddings.
3. **B. Permutation-invariant (has no built-in sense of order/position)** — Without positional info, a ViT couldn't tell an image from its scrambled patches.
4. **A. Summarize the whole image for the classification head** — Its output vector aggregates the sequence for prediction (or tokens are averaged instead).
5. **A. Attend to every other patch, giving a global receptive field in one layer** — Query-key matching mixes information globally, unlike local convolution.
6. **C. softmax(QKᵀ/√d)V — a relevance-weighted sum of values** — Query-key similarity (softmax) weights the value vectors.
7. **A. N² (quadratic in the number of patches)** — The N×N query-key matrix makes cost quadratic, driving efficient variants.
8. **C. Computing attention within shifted local windows and building a hierarchy** — Windowed, shifted attention restores locality and near-linear cost, CNN-like hierarchy.
9. **D. Underperforms, because it lacks the CNN's data-efficient inductive biases** — ViTs must learn spatial priors from data; with limited data, CNNs win.
10. **B. Measure held-out accuracy and cost for your data and budget (try both)** — Choose by measurement, not hype; the field is converging (ConvNeXt, hybrids).
