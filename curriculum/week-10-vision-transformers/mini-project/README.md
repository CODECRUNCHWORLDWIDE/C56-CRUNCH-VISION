# Mini-Project — Vision Transformer: Built, Benchmarked, and Interrogated

## Brief

Understand and benchmark the architecture reshaping vision. Build the ViT's front door yourself, run a real
one and read where it looks, and compare it honestly to a CNN under a controlled protocol — proving you can
reason about ViTs vs. CNNs with measurement, not fashion. This is the deliverable that shows you own the
Transformer applied to vision at the level of having built its input pipeline and understood its costs.

## Requirements

1. **Patch embedding from scratch.** Turn an image into a `(1, 197, D)` token sequence (patches + learned
   positional embedding + [CLS]); prove the conv-equals-Linear equivalence with an `allclose` assertion,
   visualize the patch grid, and interpolate the position table to a higher resolution.
2. **Attention from the definition.** Implement `softmax(QK^T/sqrt(d))V` and verify it against PyTorch's
   `scaled_dot_product_attention`; include the sqrt(d) ablation showing softmax saturation without the scale.
3. **Run a pretrained ViT with rollout.** Predictions on your images with correct preprocessing; overlay an
   **attention-rollout** map (accounting for residuals across layers) and contrast it with raw last-layer
   attention.
4. **ViT vs. CNN, controlled.** On a small dataset, fine-tune a ViT and a comparable CNN identically and
   report accuracy, wall-clock, and overfitting gap; then show the from-scratch collapse that exposes the
   ViT's data-hunger, and an accuracy-vs-data-fraction curve. Hold the training recipe constant, then vary
   it, to separate architecture from recipe (the ConvNeXt lesson).
5. **Cost measurement.** Plot attention cost vs. token count `N` (patch sizes 32/16/8) and confirm the `N^2`
   slope; note where the MLP vs. attention term dominates.
6. **Analysis.** A measured, hype-free write-up: for this data and budget, which wins and why, referencing
   inductive bias, data scale, transfer, and the `N^2` cost. Cite the specific evidence (ViT, DeiT,
   ConvNeXt, Swin).
7. **README.** Reproduce steps and honest limitations.

## Stretch

- A long-range-relationship task where global attention beats local convolution, with receptive-field
  arithmetic for the CNN (Challenge 1).
- Probe a self-supervised ViT: DINO emergent-segmentation overlays and a linear-probe accuracy table vs. a
  supervised backbone (Challenge 3).
- Implement windowed (Swin-style) attention with a shift and show linear cost (Exercise 4).

## What you are proving

You understand the Transformer applied to vision at the level of having built its input pipeline and
attention operation, measured its quadratic cost, and judged ViT vs. CNN by evidence — inductive bias, data
scale, and cost — rather than hype. Next week you make vision *practical*: shrinking and deploying a model to
run in real time on constrained hardware.
