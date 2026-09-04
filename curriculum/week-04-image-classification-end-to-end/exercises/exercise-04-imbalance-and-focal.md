# Exercise 4 — Long-tailed training with weighting and focal loss

**Goal:** experience the accuracy trap on imbalanced data and fix it with the tools from Lecture 5.

## Tasks

1. Construct a long-tailed version of your dataset (e.g. keep one class 10–20× rarer than the rest, or use a
   naturally imbalanced folder). Train a baseline with plain cross-entropy and show that overall accuracy looks
   fine while the rare class's recall is poor. Report per-class recall.
2. Apply and compare at least three remedies: (a) class-weighted cross-entropy (inverse frequency), (b)
   `WeightedRandomSampler` over-sampling with augmentation, and (c) focal loss with `γ = 2`. Report per-class
   recall for each.
3. **Effective number.** Replace inverse-frequency weights with the effective-number weights
   `(1−β)/(1−β^{n_c})` (Cui et al., 2019, e.g. `β = 0.999`) and report the difference.
4. Show the trade-off explicitly: improving rare-class recall usually costs some precision or head-class recall.
   Also report **balanced accuracy** (mean per-class recall) so the objective matches intent.
5. Re-check calibration (ECE) after re-balancing and temperature-scale on a distribution-matched split.

## Deliverable

A notebook with a before/after per-class recall table for each remedy, the balanced-accuracy comparison, and a
one-paragraph recommendation naming which remedy you would ship for a stated error-cost scenario, with the
trade-off it accepts.
