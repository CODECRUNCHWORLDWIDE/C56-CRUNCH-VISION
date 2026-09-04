# Challenge 1 — Beat the baseline, provably (with an interval and a test)

The capstone's minimum bar: beat your *strongest* baseline on held-out images, and *prove* the
improvement is not sampling noise.

1. Report your model and baseline metrics on the untouched test set, with the exact metric, threshold, and
   `n` stated, and a **confidence interval** on each (Wilson for accuracy; group-aware bootstrap otherwise).
2. Run a **paired significance test** of the improvement: McNemar's test for classification (report `b`,
   `c`, the χ² statistic, and the p-value) or a paired bootstrap of the metric *difference* (report the 95%
   interval of the difference and whether it excludes 0). State your conclusion in one sentence, e.g.
   "the +0.04 accuracy gain is significant (McNemar χ²=11.3, p<0.001)."
3. If you *can't* beat the strongest baseline (often a zero-shot foundation model), that is a legitimate
   outcome — analyze *why* (data too small, task too easy for the foundation model, annotations too noisy,
   wrong architecture) and document it honestly. A well-analyzed near-miss beats a fake win.

**Deliverable:** the head-to-head result with a confidence interval and a paired-test conclusion, or an
honest post-mortem. Scientific honesty about your own model — quantified, not asserted — is the skill being
tested.
