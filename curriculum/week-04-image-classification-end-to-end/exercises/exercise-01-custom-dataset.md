# Exercise 1 — Build a custom Dataset with clean splits

**Goal:** a leakage-free data pipeline from real image folders.

## Tasks

1. Take a real image-folder dataset (Oxford Pets subset, or your own labeled photos). Build a `Dataset` (or use `ImageFolder`) that loads images and labels.
2. Split into train/validation/test. If images cluster (multiple photos of one individual/breed session), split by group so no leakage occurs.
3. Apply augmented transforms to training and clean transforms to validation/test. Normalize with training-set (or ImageNet) statistics.
4. Sanity-check: display a batch of augmented training images with labels, and confirm no image appears in more than one split.

## Deliverable

A notebook with the working DataLoaders, a displayed augmented batch, and an explicit check that splits don't overlap. Two sentences on how you prevented leakage.
