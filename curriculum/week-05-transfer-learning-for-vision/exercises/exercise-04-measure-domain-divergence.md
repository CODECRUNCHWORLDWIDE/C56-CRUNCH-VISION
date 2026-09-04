# Exercise 4 — Measure the domain divergence and read the adaptation bound

**Goal:** make Lecture 4's theory concrete — estimate the H-divergence between two domains from
unlabeled features and watch fine-tuning shrink it.

## Tasks

1. Pick two datasets: a **similar** target (ImageNet-like photos, e.g. a pets or food subset) and a **different**
   target (satellite/aerial, sketches, or microscopy). With a frozen ImageNet backbone, extract features
   `phi(x)` for source-style and each target.
2. **Estimate the divergence.** Train a simple **domain classifier** (logistic regression) to distinguish
   source features from target features; its balanced accuracy above 50% is a proxy for `d_{H Delta H}`. Report it
   for the similar and the different target. Confirm the different domain is more separable (larger divergence).
3. **Shrink it.** Fine-tune the backbone on the target task, re-extract features, and re-train the domain
   classifier. Report how much the divergence dropped and whether target task accuracy rose in step — the
   Lecture-4 lever in action.
4. **Diagnose the regime.** For each target, argue whether you are in covariate-shift (small `lambda`, alignment
   helps) or concept-shift (large `lambda`, alignment cannot) territory, using the joint evidence: is the domain
   classifier separable *and* does the task transfer poorly?

## Deliverable

A notebook reporting the domain-classifier accuracy (divergence proxy) before and after fine-tuning for a
similar and a different target, a plot of divergence vs. target accuracy, and a paragraph per target naming the
shift regime and the bound term it implicates.
