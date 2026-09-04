# Exercise 1 — Patch embedding from scratch and the conv equivalence

**Goal:** turn an image into patch tokens yourself, and *prove* the convolution-equals-patch-embedding
identity in code.

## Tasks

1. Take a `(3, 224, 224)` image. Manually cut it into non-overlapping 16x16 patches with `unfold` (or a
   reshape+permute) and confirm you get `N = 196` patches, each `(3, 16, 16) = 768` values. Print the exact
   shapes at every step.
2. **Visualize** the patch grid overlaid on the image (draw the 14x14 gridlines) so the tokenization is
   concrete, and render the flattened patches as a 14x14 mosaic.
3. Implement patch embedding **two ways** and assert they produce identical tokens (up to a shared weight
   copy): (a) flatten each patch then apply `nn.Linear(768, D)`; (b) `nn.Conv2d(3, D, kernel_size=16,
   stride=16)` then `flatten(2).transpose(1,2)`. Copy the linear weight into the conv (reshaped) and check
   `torch.allclose`. This is the Lecture 1 theorem, verified.
4. Add a learned positional embedding table `(N, D)` and prepend a learnable `[CLS]` token; print the final
   sequence shape `(1, 197, D)`.
5. **Resolution interpolation.** Bicubically interpolate your `(196, D)` position table to a 24x24 grid
   `(576, D)` and confirm the shape — the exact operation needed to fine-tune a ViT at higher resolution.

## Deliverable

A notebook that turns an image into a `(1, 197, D)` token sequence, visualizes the patch grid, proves the
conv-equals-Linear equivalence with an `allclose` assertion, and interpolates the position table to a new
resolution. This is the ViT's front door, built and understood.
