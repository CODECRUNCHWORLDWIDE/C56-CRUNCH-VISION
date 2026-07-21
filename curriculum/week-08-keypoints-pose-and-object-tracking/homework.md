# Week 8 — Homework

Cement keypoints and tracking before full video understanding.

## Tasks

- Explain why heatmap regression beats direct coordinate regression for keypoints.
- Write the detect-then-associate tracking loop in pseudocode, including track birth and death.
- Describe two tracking failure modes and which metric (MOTA/IDF1/HOTA) best exposes each.
- Read a SORT/Deep SORT explainer (in resources) and note how appearance embeddings reduce ID swaps.

## Definition of done

A committed project that runs a pretrained pose estimator (drawing skeletons) and implements a multi-object tracker (detect-then-associate with IoU, and an appearance cue to reduce ID swaps) on a short clip, keeping consistent IDs, with a note on identity-switch failures and an explicit ethics/privacy statement.

Submit by committing your work to your course repo under `week-08/`.
