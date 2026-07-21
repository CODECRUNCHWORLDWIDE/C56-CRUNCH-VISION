# Exercise 3 — The single-frame baseline

**Goal:** measure whether temporal modeling actually earns its cost.

## Tasks

1. On a small action-clip set, build a **single-frame baseline**: run an image classifier on one (or a few averaged) frame(s) per clip and predict the action, aggregating if needed.
2. Compare its accuracy to the temporal model from Exercise 2 on the same clips.
3. Identify which actions the single-frame baseline gets right (visible in one frame) and which need temporal context (motion-defined).
4. Estimate the compute cost difference between the two approaches.

## Deliverable

A notebook comparing single-frame vs. temporal accuracy per action, with a note on which actions truly needed time and whether the temporal model's cost was justified. This baseline is the honesty check for all video work.
