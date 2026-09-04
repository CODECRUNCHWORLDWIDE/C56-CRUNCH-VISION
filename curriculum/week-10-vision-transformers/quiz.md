# Week 10 — Quiz

Fifteen questions spanning the patch-embedding pipeline, attention's algebra and cost, ViT-vs-CNN inductive bias and scaling, the efficient/hierarchical zoo, and the self-supervised/multimodal frontier. Attempt each before the answer key.

**1. A Vision Transformer's first step turns an image into:**

- A. a sequence of patch tokens, each a linearly embedded flattened patch
- B. a single global feature vector
- C. a pixel-wise segmentation mask
- D. a set of bounding boxes

<details>
<summary>Answer</summary>

**A. a sequence of patch tokens, each a linearly embedded flattened patch** — The ViT chops the image into non-overlapping patches and linearly projects each into a token — the image-as-sequence reframe.

</details>

**2. The claim 'patch embedding equals a non-overlapping stride-P convolution' holds because:**

- A. convolution is always equivalent to a linear layer
- B. the softmax makes them equal
- C. positional embeddings force the equivalence
- D. with stride = kernel = P the receptive fields tile the image disjointly, so each output is a per-patch linear projection E·x_p + b

<details>
<summary>Answer</summary>

**D. with stride = kernel = P the receptive fields tile the image disjointly, so each output is a per-patch linear projection E·x_p + b** — Stride equal to kernel size gives disjoint tiling patches; flattening the kernel per output channel reproduces the shared linear map exactly.

</details>

**3. Positional embeddings are required because self-attention is:**

- A. too slow without them
- B. non-differentiable
- C. permutation-equivariant — permuting tokens permutes outputs identically, so raw attention has no notion of position
- D. defined only for text

<details>
<summary>Answer</summary>

**C. permutation-equivariant — permuting tokens permutes outputs identically, so raw attention has no notion of position** — Attention(Pi X) = Pi Attention(X); without a position code a ViT cannot distinguish an image from its shuffled patches.

</details>

**4. Scaled dot-product attention divides the logits by sqrt(d_k) in order to:**

- A. keep the logit variance ~1 so softmax stays well-conditioned and gradients do not vanish
- B. make the matrix square
- C. enforce permutation invariance
- D. reduce the FLOP count

<details>
<summary>Answer</summary>

**A. keep the logit variance ~1 so softmax stays well-conditioned and gradients do not vanish** — A dot of two unit-variance d_k-vectors has variance d_k; dividing by sqrt(d_k) renormalizes it, preventing softmax saturation.

</details>

**5. Self-attention's time and memory cost scales as:**

- A. Theta(N log N) time
- B. Theta(N^2 D) time and Theta(N^2) memory, quadratic in the token count N
- C. constant in N
- D. Theta(N D^2) time, Theta(N) memory

<details>
<summary>Answer</summary>

**B. Theta(N^2 D) time and Theta(N^2) memory, quadratic in the token count N** — Forming and applying the N×N score matrix costs Theta(N^2 D) time and Theta(N^2) memory — the quadratic barrier.

</details>

**6. Halving the patch size (16 -> 8) at fixed resolution multiplies attention cost by roughly:**

- A. 16x
- B. 2x
- C. 4x
- D. 8x

<details>
<summary>Answer</summary>

**A. 16x** — Halving patch size quadruples N; since cost grows as N^2, it rises about 4^2 = 16x.

</details>

**7. FlashAttention improves attention by:**

- A. replacing softmax with a linear kernel
- B. IO-aware tiling that avoids materializing the N×N matrix in HBM, cutting memory to O(N) with large wall-clock wins
- C. removing the value projection
- D. reducing the FLOP count below N^2

<details>
<summary>Answer</summary>

**B. IO-aware tiling that avoids materializing the N×N matrix in HBM, cutting memory to O(N) with large wall-clock wins** — FlashAttention keeps the exact N^2 FLOPs but tiles the computation to avoid storing the full score matrix, saving memory and time.

</details>

**8. The original ViT trained from scratch on ImageNet-1k, versus a comparable ResNet, typically:**

- A. underperforms, because it lacks the CNN's data-efficient inductive biases (locality, translation equivariance)
- B. always wins by a large margin
- C. cannot train at all
- D. is exactly identical

<details>
<summary>Answer</summary>

**A. underperforms, because it lacks the CNN's data-efficient inductive biases (locality, translation equivariance)** — With fewer built-in priors the ViT must learn spatial structure from data; on limited data the CNN's biases win.

</details>

**9. DeiT closed the ImageNet-1k gap without JFT-300M mainly by:**

- A. using a larger patch size
- B. removing self-attention
- C. training for fewer epochs
- D. heavy augmentation, strong regularization, and a distillation token that imports a CNN teacher's inductive bias

<details>
<summary>Answer</summary>

**D. heavy augmentation, strong regularization, and a distillation token that imports a CNN teacher's inductive bias** — Touvron et al. (2021) used RandAugment/Mixup/CutMix and a distillation token learning from a CNN teacher — data-efficiency via imported bias.

</details>

**10. ConvNeXt's main lesson for ViT-vs-CNN comparisons is that:**

- A. CNNs can never match ViTs
- B. much of the apparent ViT advantage was the modern training recipe, not attention itself — a pure CNN with the same recipe matches ViTs
- C. attention is unnecessary for language too
- D. ViTs are always faster

<details>
<summary>Answer</summary>

**B. much of the apparent ViT advantage was the modern training recipe, not attention itself — a pure CNN with the same recipe matches ViTs** — Liu et al. (2022) modernized a ResNet's recipe and matched ViTs, showing recipe must be held constant for a fair architectural comparison.

</details>

**11. Swin Transformer makes attention scale (near-)linearly in N by:**

- A. removing positional information
- B. computing attention within local M×M windows and shifting the window partition every other layer to exchange information across borders
- C. deleting the MLP blocks
- D. using a single attention head

<details>
<summary>Answer</summary>

**B. computing attention within local M×M windows and shifting the window partition every other layer to exchange information across borders** — Windowed attention is Theta(N M^2 D) (linear in N); the shifted-window trick restores cross-window flow, and patch-merging builds a CNN-like hierarchy.

</details>

**12. Attention rollout (Abnar & Zuidema, 2020) gives a cleaner saliency map than a single layer because it:**

- A. uses the largest logit only
- B. accounts for residual connections and composes attention across all layers via matrix products
- C. ignores the CLS token
- D. requires ground-truth masks

<details>
<summary>Answer</summary>

**B. accounts for residual connections and composes attention across all layers via matrix products** — Rollout adds the residual (0.5A + 0.5I), renormalizes, and multiplies down the stack, approximating total information flow rather than one layer.

</details>

**13. MAE (He et al., 2022) uses a mask ratio around 75% — much higher than BERT's 15% — because:**

- A. image patches are highly redundant, so low masking makes reconstruction trivially solvable by interpolation and teaches little semantics
- B. images have fewer pixels than sentences have words
- C. GPUs require it
- D. the decoder cannot handle more tokens

<details>
<summary>Answer</summary>

**A. image patches are highly redundant, so low masking makes reconstruction trivially solvable by interpolation and teaches little semantics** — High redundancy means a hard, semantic pretext task requires masking most patches; the encoder also only sees the visible 25%, making pretraining cheaper.

</details>

**14. A striking emergent property of DINO-trained ViTs (Caron et al., 2021) is that:**

- A. their [CLS]-token attention maps segment the salient object with no segmentation labels
- B. they run in linear time
- C. they require no augmentation
- D. they need no positional embeddings

<details>
<summary>Answer</summary>

**A. their [CLS]-token attention maps segment the salient object with no segmentation labels** — Self-distillation across crops teaches the model what belongs together so well that attention localizes objects for free — absent in supervised ViTs.

</details>

**15. CLIP (Radford et al., 2021) enables zero-shot classification because its objective:**

- A. distills from a CNN teacher
- B. predicts pixel values of masked patches
- C. contrastively aligns image and text embeddings on web-scale pairs, so classes can be specified by natural-language prompts at test time
- D. classifies among a fixed 1000-class softmax

<details>
<summary>Answer</summary>

**C. contrastively aligns image and text embeddings on web-scale pairs, so classes can be specified by natural-language prompts at test time** — CLIP's InfoNCE contrast over an image-text similarity matrix makes language the label space, so 'a photo of a {class}' prompts classify unseen categories.

</details>

---
