# Challenge 1 — Tame an imbalanced dataset and defend the remedy

Real data is imbalanced and the "best" fix depends on what a mistake costs. Build a classifier that
handles imbalance honestly and justify your choice quantitatively.

**Part 1 — expose the trap.** Create or find an imbalanced dataset (one class ~10× rarer). Train a naive
classifier and show that overall accuracy looks fine while the rare class's recall is poor. Report per-class
recall *and* the accuracy confidence interval so the reader knows the noise floor.

**Part 2 — compare remedies.** Apply and compare at least three: class-weighted loss, `WeightedRandomSampler`
over-sampling (with augmentation), focal loss (`γ = 2`), and — for ambition — logit adjustment by `log p(y)`
(Menon et al., 2021). Report per-class recall before and after, plus balanced accuracy.

**Part 3 — the trade-off and the decision.** Show the cost: improving rare-class recall usually reduces
head-class recall or precision. State a concrete deployment scenario with an explicit cost per error type, pick
the remedy you would ship, and defend it in those terms. Re-check calibration after re-balancing.

**Deliverable:** a before/after per-class metric table, a balanced-accuracy comparison, and a justified
recommendation tied to a stated error-cost model. The lesson — accuracy hides imbalance, and the right fix
depends on what a mistake actually costs — is one you will use in every real project.
