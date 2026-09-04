# Week 10 — Homework

Cement the ViT — its front door, its attention algebra and cost, its inductive-bias trade-offs, and its self-supervised frontier — before making vision deployable next week. Do the derivations by hand before coding; the sqrt(d) and N^2 arguments should be muscle memory, not incantations.

## Tasks

- Explain, in your own words, how an image becomes a token sequence, and prove (in a few lines) that a stride-P non-overlapping convolution computes the patch embedding.
- Derive why attention cost is Theta(N^2 D) in time and Theta(N^2) in memory, and compute exactly how much cost rises when you halve the patch size at fixed resolution.
- Give the variance argument for the 1/sqrt(d) scale in scaled dot-product attention, and state what fails in training without it.
- Summarize the ViT-vs-CNN choice as a function of data scale, transfer, resolution, and edge deployment, citing the ViT, DeiT, and ConvNeXt evidence explicitly.
- Read the ViT and ConvNeXt abstracts and the MAE and DINO abstracts; write one paragraph each on what each says about architecture vs. training recipe, and what self-supervision buys.
- Extend your mini-project to add an attention-rollout overlay and confirm it localizes the object better than raw last-layer attention.

## Definition of done

A committed notebook and README that: implement patch embedding from scratch with a proven conv-equals-Linear equivalence and a `(1, 197, D)` sequence; implement and verify scaled dot-product attention with the sqrt(d) ablation; run a pretrained ViT with an attention-rollout overlay; compare ViT vs. CNN on a small dataset under an identical, then varied, training recipe (fine-tune and from-scratch) with accuracy, time, gap, and a data-fraction curve; plot the N^2 cost slope; and close with a measured, hype-free analysis citing ViT/DeiT/ConvNeXt/Swin evidence.

Submit by committing your work to your course repo under `week-10/`.
