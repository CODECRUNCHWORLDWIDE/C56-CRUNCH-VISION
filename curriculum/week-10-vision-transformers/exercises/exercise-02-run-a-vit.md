# Exercise 2 — Run a pretrained ViT and compute attention rollout

**Goal:** use a real ViT and read what it attends to with a *correct* attention visualization.

## Tasks

1. Load a pretrained ViT (torchvision `vit_b_16`, or a `timm` model like `vit_base_patch16_224`) with its
   exact preprocessing transform. Run it on several of your images and report the top-5 predictions with
   softmax confidences.
2. **Hook the attention.** Register forward hooks to capture each block's attention matrix
   `A_l in R^{(N+1) x (N+1)}` (average over heads). Confirm each row sums to 1.
3. **Attention rollout** (Abnar & Zuidema, 2020, Lecture 4). Add the residual and renormalize each layer,
   `hat{A}_l = normalize(0.5 A_l + 0.5 I)`, then multiply down the stack. Take the CLS row, drop the CLS
   entry, reshape to 14x14, upsample to the image, and overlay as a heatmap. Does it localize the object?
4. Compare rollout against **raw last-layer** CLS attention on the same image — show that rollout is cleaner
   because it accounts for residual mixing across layers.
5. Compare the ViT's predictions to a pretrained CNN (e.g. ResNet-50) on the same images; note agreements
   and disagreements.

## Deliverable

A notebook with ViT top-5 predictions, an **attention-rollout overlay** contrasted against raw last-layer
attention, and a ViT-vs-CNN prediction comparison. Seeing the rollout focus on the object — and seeing why
rollout beats a single layer — demystifies the model.
