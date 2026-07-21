# Exercise 1 — Implement patch embedding

**Goal:** turn an image into patch tokens yourself.

## Tasks

1. Take a 224×224 image. Manually cut it into non-overlapping 16×16 patches and confirm you get 196 patches, each 16×16×3.
2. Visualize the patch grid overlaid on the image so the tokenization is concrete.
3. Implement patch embedding two ways and confirm they match: (a) reshape/flatten patches then a `Linear` layer, and (b) a `Conv2d(3, dim, kernel_size=16, stride=16)`. Verify both give `(N, 196, dim)`.
4. Add a learned positional embedding and a [CLS] token; print the final sequence shape `(N, 197, dim)`.

## Deliverable

A notebook that turns an image into a `(N, 197, dim)` token sequence, with the patch grid visualized and the conv-equals-patch-embed equivalence confirmed. This is the ViT's front door.
