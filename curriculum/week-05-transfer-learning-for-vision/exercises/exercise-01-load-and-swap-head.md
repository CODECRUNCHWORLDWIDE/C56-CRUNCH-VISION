# Exercise 1 — Load a backbone, swap the head, and get the preprocessing right

**Goal:** the mechanics of reusing a pretrained network, with the preprocessing correctness that most
tutorials skip.

## Tasks

1. Load a pretrained backbone from torchvision (ResNet-50 with `ResNet50_Weights.IMAGENET1K_V2`, or a `timm`
   model). Print its module tree and identify the **backbone** (feature layers) and the **head** (final `fc`/
   `classifier`). Print the head's `in_features`.
2. Replace the head with a new `nn.Linear` (or a small MLP with dropout) sized to your dataset's number of
   classes. Confirm the new head's parameters are the only ones with freshly-random weights.
3. **Preprocessing, done right.** Obtain the backbone's exact preprocessing from `weights.transforms()` and
   *print it*. Manually confirm the normalization mean/std and input size. Then deliberately build a *wrong*
   transform (e.g. no normalization, or ImageNet stats on a non-ImageNet backbone) and show, on one batch, that
   the feature vectors differ substantially — the silent failure from Lecture 3.
4. Run one forward pass on a correctly-preprocessed batch and confirm output shape `(N, num_classes)`.
5. **Cache features.** Run the frozen backbone over your training set once and cache `phi(x)` to disk; report
   the shape and how much faster a subsequent "epoch" over cached features is than a forward pass through the
   full backbone.

## Deliverable

A notebook that loads the backbone, swaps the head, prints and verifies the exact preprocessing (with the
wrong-vs-right feature comparison), produces correctly-shaped outputs, and caches features — the foundation for
both strategies.
