# Exercise 3 — Reduce identity switches with appearance (toward Deep SORT)

**Goal:** see tracking's signature failure and mitigate it with an appearance embedding and
Mahalanobis gating.

## Tasks

1. Run your Exercise 2 tracker on a clip where two same-class objects **cross or occlude**. Watch for ID
   swaps at the crossing and count them (by eye or against a few hand-labelled frames). Confirm IoU-only
   association swaps IDs when boxes overlap.
2. **Add appearance.** For each detection compute an embedding — a colour histogram, or (better) features
   from a pretrained backbone crop. Store a small gallery of recent embeddings per track.
3. **Fuse costs.** Replace the pure-IoU cost with `c = lambda * d_mahalanobis + (1 - lambda) * d_cosine`,
   where `d_mahalanobis` uses your Kalman innovation covariance `S` (gate at the chi-squared 0.95 quantile,
   `d^2 < 9.49` for 4-D) and `d_cosine` is the smallest cosine distance to the track's gallery. Sweep
   `lambda`.
4. Show the appearance-fused cost reduces ID swaps at the crossing versus IoU-only, and report the count
   before/after.

## Deliverable

A before/after comparison of ID swaps at a crossing, with and without the appearance + Mahalanobis cost, the
swap counts, and a note on why position alone is insufficient. This is the exact problem Deep SORT solved
(Wojke et al., 2017).
