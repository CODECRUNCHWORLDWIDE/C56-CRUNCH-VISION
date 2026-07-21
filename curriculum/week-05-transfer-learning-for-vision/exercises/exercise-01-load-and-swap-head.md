# Exercise 1 — Load a backbone and adapt its head

**Goal:** the mechanics of reusing a pretrained network.

## Tasks

1. Load a pretrained ResNet-18 (or MobileNetV2) from torchvision with ImageNet weights.
2. Inspect its structure — identify the backbone (feature layers) and the head (final `fc`/classifier). Print the head's input feature count.
3. Replace the head with a new `Linear` (or small MLP) sized to your dataset's number of classes.
4. Run one forward pass on a correctly-preprocessed batch and confirm the output has shape `(N, num_classes)`.

## Deliverable

A notebook that loads the backbone, swaps the head, and produces correctly-shaped outputs on a preprocessed batch. Confirm you used the backbone's exact normalization — print the transform. This is the foundation for both strategies.
