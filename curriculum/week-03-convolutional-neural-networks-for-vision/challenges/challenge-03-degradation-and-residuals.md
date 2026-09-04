# Challenge 3 — Reproduce the degradation problem and fix it with residuals

This is a research-flavored challenge that reproduces the central experiment of He et al. (2016).
The claim to test: past some depth, *plain* deep networks get **worse training error** — an optimization
failure, not overfitting — and residual connections fix it.

**Setup.** On CIFAR-10, build a family of plain CNNs of increasing depth (e.g. 8, 14, 20, 32 conv layers)
using identical building blocks, BN, and the same training recipe. Then build a matched family of **residual**
networks — same blocks, but each pair wrapped with a skip connection `y = F(x) + x` (use a 1x1 conv on the
shortcut when channel counts change).

**Investigate.**
1. Plot **training** accuracy vs. depth for both families. Do the plain nets degrade past some depth while
   the residual nets keep improving (or at least not degrade)? Report *training* accuracy specifically — the
   degradation problem is about optimization, so do not confound it with test/overfitting behavior.
2. For one deep plain net and its residual twin, plot the training-loss curves together. Does the residual
   version optimize faster and reach lower training loss?
3. Inspect gradient magnitudes at the earliest layers of a deep plain net vs. the residual one over training.
   Relate what you see to the `dy/dx = dF/dx + I` argument from Lecture 5.

**Analyze.** Explain your findings in terms of (i) identity mappings being easy to represent with a skip and
(ii) the `+I` gradient highway. State clearly where your small-scale experiment agrees with He et al. and
where it cannot (compute, depth, and dataset scale differ).

**Deliverable:** a short report with the depth-vs-training-accuracy plots for both families, the paired
loss curves, and an honest account of what your experiment does and does not establish. A well-analyzed
partial or negative result earns full credit — the graded skill is experimental rigor and interpretation.
