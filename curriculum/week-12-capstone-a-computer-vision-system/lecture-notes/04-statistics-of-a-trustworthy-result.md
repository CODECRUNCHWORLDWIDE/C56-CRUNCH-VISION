# Lecture 4 — The statistics of a trustworthy result: intervals, significance, and proper scoring

Your headline metric is a **random variable**. Measure accuracy on a different 500-image test
set drawn from the same distribution and you get a different number. The undergraduate capstone reports the
one number it happened to get; the graduate capstone reports the number *with an interval*, and proves any
claimed improvement over the baseline is unlikely to be sampling noise. This lecture gives you the tools —
and the derivations — to do that.

## Accuracy as a binomial estimate

Model your test accuracy as the mean of `n` independent Bernoulli trials: each of the `n` test images is
either correct (1) or not (0), with unknown success probability `p`. The observed accuracy `p̂ = k/n` is the
maximum-likelihood estimate of `p`, and its sampling variance is `Var(p̂) = p(1−p)/n`. The naive **Wald**
interval `p̂ ± z·√(p̂(1−p̂)/n)` is taught everywhere and is *wrong* near 0 or 1 and for small `n` — it can
even extend past [0,1]. Prefer the **Wilson score interval**, which inverts the score test and stays inside
[0,1]:

    (p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² )) / (1 + z²/n).

Worked example: 440 correct of 500, `p̂ = 0.88`. Wald gives `0.88 ± 1.96·√(0.88·0.12/500) = 0.88 ± 0.0285`,
i.e. [0.851, 0.909]. The Wilson interval is slightly asymmetric and marginally tighter but, crucially,
well-behaved — use it, and *always report the `n` behind the estimate*, because an interval width of ±0.03
tells a reviewer more than three decimal places of false precision. (Brown, Cai & DasGupta, 2001,
"Interval Estimation for a Binomial Proportion", *Statistical Science*, is the definitive comparison.)

## The bootstrap: intervals for metrics with no closed form

mAP, mIoU, F1, and AUC have no clean binomial variance. For these, use the **nonparametric bootstrap**
(Efron, 1979, "Bootstrap Methods", *Annals of Statistics*). Treat your test set of `n` items as the
population; resample `n` items *with replacement* `B` times (B = 2,000–10,000); recompute the full metric on
each resample; the 2.5th and 97.5th percentiles of those `B` values form a 95% percentile interval. The
bootstrap works because the empirical distribution is a consistent estimate of the true one, so resampling
from it approximates resampling from the world. Two vision-specific cautions: (1) bootstrap the *right unit*
— if your test images are grouped (frames from one video, crops from one scan), resample **groups**, not
images, or you understate the variance exactly as leakage does; (2) for detection/segmentation, resample
whole images and recompute the metric end to end, never per-box.

## Proving you beat the baseline: paired tests

Comparing your model to the baseline on the *same* test set is a **paired** comparison, and pairing is
statistical gold — it removes the image-to-image difficulty variance and tests only the model difference.

- **Classification: McNemar's test.** Build the 2×2 table of *disagreements*: `b` = images your model got
  right but the baseline got wrong, `c` = the reverse. Images both got right or both wrong carry no
  information about which is better and drop out. Under the null "the two models are equally accurate,"
  `b` and `c` are exchangeable, so `b ~ Binomial(b+c, 0.5)`. The test statistic
  `χ² = (|b − c| − 1)² / (b + c)` (with continuity correction) is compared to χ² with 1 d.f. (McNemar,
  1947; Dietterich, 1998, "Approximate Statistical Tests for Comparing Supervised Classification Learning
  Algorithms", *Neural Computation*, recommends it precisely for this setting). Intuition: if the models
  were equal, the off-diagonal should split roughly 50/50; a lopsided split is evidence of a real
  difference.
- **Anything else: the paired bootstrap.** For each of `B` resamples, compute `metric_model − metric_baseline`
  on the *same* resampled images. If the 95% interval of that *difference* excludes 0, your improvement is
  significant at α=0.05. This is the general-purpose tool and it works for mAP, mIoU, F1, and AUC alike.

A capstone that reports "we improved from 0.84 to 0.88" without one of these tests has not shown the
improvement is real — with a 500-image test set that gap can easily be noise.

## Proper scoring rules: why accuracy isn't enough

Accuracy throws away the confidence. A **proper scoring rule** is a loss on *probabilistic* predictions,
minimized in expectation only by reporting the true probabilities — it rewards being both correct *and*
honestly uncertain. The two canonical ones:

- **Negative log-likelihood** (cross-entropy): `−(1/n)Σ log p(y_i)`. Punishes confident wrong predictions
  savagely (a confident miss sends the log to −∞), which is exactly why it exposes the overconfidence of
  Lecture 2.
- **Brier score**: `(1/n)Σ ‖p_i − onehot(y_i)‖²`, the mean squared error of the probability vector. Gneiting
  & Raftery (2007, "Strictly Proper Scoring Rules", *JASA*) show the Brier score decomposes cleanly into
  *calibration* + *refinement* terms, tying it directly to the reliability diagram of Lecture 2.

Reporting NLL or Brier alongside accuracy tells a reviewer whether your model's *probabilities* are
trustworthy, not just its top-1 label — and for any system that acts on confidence, that is the number that
matters.

## Multiple comparisons and the garden of forking paths

If you try twenty architectures and report the best on the test set, your best number is biased upward — you
have implicitly done twenty tests and kept the luckiest (the multiple-comparisons problem). Guard against it:
select on *validation*, report the single chosen model on the test set *once*, and if you must compare
several final models, adjust (Bonferroni: divide α by the number of comparisons; or Benjamini–Hochberg to
control the false-discovery rate). Naming this in your defense signals statistical maturity.

**Takeaway:** treat every metric as an estimate with a sampling distribution. Put a Wilson interval on
accuracy and a (group-aware) bootstrap interval on mAP/mIoU/F1; prove your win over the baseline with
McNemar (classification) or a paired bootstrap of the *difference* (everything else); report a proper
scoring rule (NLL or Brier) so the probabilities are judged, not just the labels; and account for
multiple comparisons so your best number isn't just your luckiest. A result without an interval is a
rumor.
