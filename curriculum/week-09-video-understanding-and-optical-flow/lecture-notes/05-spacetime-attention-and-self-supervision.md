# Lecture 5 — Spacetime self-attention & self-supervised video

The frontier of video understanding is (a) **self-attention over spacetime**, which replaces
convolution's local temporal receptive field with global, learnable temporal reach, and (b)
**self-supervised pretraining**, which sidesteps video's crippling label scarcity. This lecture makes
both precise, including the cost arithmetic that forces the key design decision.

## Tokens in spacetime, and why cost explodes

A Vision Transformer (Week 10) splits an image into `N` patches ("tokens") and applies self-attention,
whose cost is **quadratic** in the number of tokens: `O(N²·d)`. A video of `T` frames has roughly `N·T`
spacetime tokens, so *joint* space-time attention costs `O((N·T)²·d)`. Doubling either the resolution
(more `N`) or the clip length (more `T`) quadruples compute. This quadratic blow-up is *the* problem
video Transformers must engineer around.

## Factorizing the attention

TimeSformer (Bertasius et al. 2021) studied several factorizations and found **divided space-time
attention** the best accuracy/cost trade:

- **Spatial attention:** each token attends only to tokens in the *same frame* — cost `O(T·N²)`.
- **Temporal attention:** each token attends only to tokens at the *same spatial location across frames*
  — cost `O(N·T²)`.

Applying the two in sequence (per block) reduces cost from `O((NT)²)` to `O(TN² + NT²)` — a large saving
when both `N` and `T` are big — while still allowing information to flow across all of spacetime over
stacked blocks. The paper's title asks the question directly: *"Is space-time attention all you need for
video understanding?"* — and answers largely yes, given enough pretraining data. ViViT (Arnab et al.
2021) explores tubelet embeddings and factorized encoders; Video Swin (Liu et al. 2022) restores
locality with **shifted 3-D windows**, trading some global reach for linear-in-tokens cost and strong
efficiency — the SlowFast-vs-Transformer tension of the field.

## Inductive bias vs. data

Convolutions bake in **locality and translation equivariance**; Transformers do not, so they need more
data to *learn* those biases — which is why video Transformers lean hard on large-scale pretraining
(image pretraining via inflated/tubelet init, or huge video corpora). The trade is the same one Week 10
draws for images: less built-in bias, more capacity, more data hunger.

## Self-supervised video pretraining

Labels are the bottleneck, so the frontier pretrains on **unlabeled** video and fine-tunes on small
labeled sets. Three families:

- **Contrastive.** Treat two clips from the same video (or the same clip augmented) as positives and
  clips from other videos as negatives; pull positives together, push negatives apart. Temporal
  augmentations (different speeds, temporal crops) build motion invariance. CVRL and related methods
  extend SimCLR/MoCo to video.
- **Masked autoencoding.** VideoMAE (Tong et al. 2022) masks a *very high* fraction (~90-95%) of
  spacetime patches and trains the model to reconstruct them. High masking works because video is
  extremely redundant across time (adjacent frames are near-copies) — a low mask ratio would let the
  model cheat by copying neighbors. This yields strong action-recognition backbones with no labels.
- **Pretext tasks.** Predict frame order, arrow-of-time (is the clip playing forward or backward?),
  playback speed, or future frames — each forcing the model to learn temporal structure. Historically
  important (Misra et al. 2016, shuffle-and-learn), now largely superseded by contrastive/MAE.

The through-line: exploit video's own temporal redundancy and coherence as the free supervisory signal.

## Where this meets ethics again

Bigger self-supervised video models trained on web-scraped footage inherit *and amplify* the
demographic and geographic bias of what is on the web, and they lower the cost of building behavior
surveillance. The heightened-privacy stance of Lecture 3 applies with more force as capability grows:
capability is not consent.

**Takeaway:** spacetime self-attention gives video Transformers global temporal reach, but joint
attention is quadratic in `N·T` spacetime tokens, so practical models **factorize** into divided
space-time attention (TimeSformer, `O(TN²+NT²)`) or windowed 3-D attention (Video Swin). Transformers
trade convolution's built-in locality for capacity and data hunger, met by **self-supervised**
pretraining — contrastive learning and, especially, high-ratio masked autoencoding (VideoMAE) that
exploits video's temporal redundancy to learn without labels. Growing capability raises, not lowers,
the surveillance-ethics bar.
