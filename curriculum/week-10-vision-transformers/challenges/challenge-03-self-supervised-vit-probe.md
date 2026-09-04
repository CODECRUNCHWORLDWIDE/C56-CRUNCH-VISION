# Challenge 3 — Probe a self-supervised ViT (DINO/MAE) and read its emergent structure

This is an open, frontier-flavored challenge (Lecture 5). Self-supervised ViTs learn structure with
no labels; your job is to *observe* that structure and evaluate the representation honestly.

1. **Emergent segmentation.** Load a pretrained **DINO** (or DINOv2) ViT. For several images, extract the
   [CLS]-token attention from the last block, reshape to the patch grid, and overlay it. Does it localize
   the salient object without any segmentation training (Caron et al., 2021)? Compare against a *supervised*
   ViT's attention on the same images and characterize the difference.
2. **Linear-probe evaluation.** Freeze the SSL backbone, extract features for a small labeled dataset, and
   train only a linear classifier on top. Compare the linear-probe accuracy of a DINO/MAE backbone against a
   supervised-pretrained ViT and a supervised CNN. This is the standard, honest way to judge a representation
   — downstream accuracy, not reconstruction sharpness.
3. **(If feasible) Masking sensitivity.** For an MAE-pretrained model, reconstruct images at mask ratios
   {25%, 50%, 75%, 90%} and discuss why high masking (75%) is what makes the pretext task teach semantics
   rather than interpolation (He et al., 2022).
4. **Optional — registers.** If your ViT is large, inspect the per-token attention norms and look for the
   high-norm artifact tokens Darcet et al. (2024) describe; discuss whether register tokens would help.

**Deliverable:** a report with DINO-vs-supervised attention overlays, a linear-probe accuracy table (SSL vs.
supervised), and an honest discussion of what self-supervision buys and what it costs. Negative or partial
results, well-analyzed, earn full credit — the graded skill is observation and interpretation, not
confirming a headline.
