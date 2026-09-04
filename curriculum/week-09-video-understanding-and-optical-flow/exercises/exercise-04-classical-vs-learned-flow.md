# Exercise 4 — Classical vs. learned flow, and warping quality

**Goal:** feel *why* learned flow (RAFT) exists by pitting it against classical Farneback on
hard footage, and measure flow quality without ground truth via warping error.

## Tasks

1. On the same clips, compute dense flow two ways: classical **Farneback** and a learned model —
   **RAFT** (`torchvision.models.optical_flow.raft_large` with pretrained weights) or PWC-Net.
2. Render both color-coded fields side by side. Note qualitative differences on **large motion**,
   **occlusion boundaries**, and **low-texture regions** (Lecture 4).
3. Since you lack ground-truth flow, use a **proxy metric**: **warp** frame `t+1` back toward frame `t`
   using each estimated flow and measure the photometric residual (mean absolute error) in
   non-occluded regions. Lower residual = better flow (where brightness constancy holds).
4. Find a clip where classical flow clearly fails and learned flow succeeds (fast motion or occlusion),
   and one where they are comparable (slow, textured motion). Explain both outcomes with Lecture 1/4
   theory.
5. Report the compute cost: RAFT needs a GPU and is far heavier than Farneback — quantify the
   accuracy-vs-cost trade.

## Deliverable

A notebook with side-by-side flow visualizations, the warping-residual comparison table, at least one
clear win for learned flow with a theory-grounded explanation, and a note on the cost trade. Deliverable:
you can justify when a learned flow model is worth its cost.
