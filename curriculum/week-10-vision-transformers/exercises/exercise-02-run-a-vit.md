# Exercise 2 — Run a pretrained ViT and inspect attention

**Goal:** use a real ViT and peek at what it attends to.

## Tasks

1. Load a pretrained ViT (torchvision `vit_b_16`, or a `timm` model) with its correct preprocessing.
2. Run it on several of your images and report the top predictions with confidences.
3. If accessible, extract and visualize an **attention map** — which patches the [CLS] token attends to for its prediction — overlaid on the image. Does it focus on the object?
4. Compare its predictions to a pretrained CNN (e.g. ResNet) on the same images; note agreements and disagreements.

## Deliverable

A notebook with ViT predictions, an attention-map overlay (if available), and a ViT-vs-CNN prediction comparison on your images. Seeing the attention focus on the object demystifies the model.
