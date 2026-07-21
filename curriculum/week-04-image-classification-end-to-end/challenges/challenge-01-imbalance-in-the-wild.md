# Challenge 1 — Tame an imbalanced dataset

Real data is imbalanced. Build a classifier that handles it honestly.

1. Create or find an imbalanced dataset (e.g. one class 10× rarer than the others). Train a naive classifier and show that overall accuracy looks fine while the rare class's recall is poor.
2. Apply and compare at least two remedies: a class-weighted loss, oversampling with `WeightedRandomSampler`, and/or targeted augmentation of the rare class.
3. Report per-class recall before and after. Show the trade-off — improving rare-class recall often costs some precision or majority-class recall.
4. Decide which remedy you'd ship and justify it in terms of the task's real cost of errors.

**Deliverable:** a before/after per-class metric table and a justified recommendation. The lesson — accuracy hides imbalance, and the right fix depends on what a mistake actually costs — is one you'll use in every real project.
