# Challenge 3 — Calibration, abstention, and reporting under shift (open)

This is an open, research-flavored challenge about *honest* classification: making the model's
probabilities trustworthy and reporting fairly when the deployment distribution may differ from your test set.

**Setup.** Train a solid classifier on your dataset. Build a reliability diagram and compute ECE; you will
almost certainly find over-confidence (Guo et al., 2017).

**Part 1 — calibrate.** Fit temperature scaling on the validation set and report ECE before/after, confirming
accuracy is unchanged. Compare against at least one alternative (Platt scaling, isotonic regression, or label
smoothing during training) and discuss which you would trust and why.

**Part 2 — abstention.** Use the calibrated confidence to build a *reject option*: abstain when confidence is
below a threshold `τ`. Plot the accuracy-vs-coverage (risk–coverage) curve: as you abstain on more low-
confidence inputs, accuracy on the rest should rise. State the coverage at which you reach a target accuracy,
and argue when abstention is appropriate (medical triage) and when it is not.

**Part 3 — shift and subgroups.** Construct a shifted test set (different class proportions, or a held-out
camera/source) and show how the reported accuracy and ECE move. If your data has an identifiable subgroup
attribute, disaggregate metrics by subgroup (cf. Buolamwini & Gebru, 2018) and report the disparity.

**Deliverable:** a short report with the reliability diagrams, the risk–coverage curve, the shifted-set and
subgroup results, and an honest account of what your model's confidence does and does not justify. Well-analyzed
negative results earn full credit — the graded skill is rigor and honesty, not a headline number.
