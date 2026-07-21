# Week 10 — Homework

Cement the ViT before making vision deployable.

## Tasks

- Explain, in your own words, how an image becomes a token sequence and why positional embeddings are required.
- Derive why attention cost is quadratic in patch count and what that means for high resolution.
- Summarize the four-quadrant-style guidance for choosing a ViT vs. a CNN (data scale, transfer, resolution, edge).
- Read the ViT and ConvNeXt paper abstracts (in resources) and note what each says about architecture vs. training recipe.

## Definition of done

A committed project that implements patch embedding from scratch (image → (N, 197, dim) tokens with positional + CLS), runs a pretrained ViT on your images (ideally with an attention visualization), and rigorously compares ViT vs. CNN on a small dataset (fine-tuned and/or from scratch) with a measured, hype-free analysis of the trade-offs.

Submit by committing your work to your course repo under `week-10/`.
