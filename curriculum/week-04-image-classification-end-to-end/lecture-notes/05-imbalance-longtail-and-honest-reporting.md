# Lecture 5 — The long tail: imbalanced learning, focal loss, and reporting under shift

Benchmark datasets are curated to be balanced; real datasets are **long-tailed** — a few head
classes dominate and a long tail of rare classes has a handful of examples each (Van Horn & Perona, 2017, "The
Devil is in the Tails"). Naive training on such data optimizes overall accuracy by ignoring the tail, producing
a model that is confidently useless exactly where you needed it. This lecture is the specialization: how to
train and, just as importantly, how to *report* honestly when the data is imbalanced and the deployment
distribution may differ from the test set.

## Why cross-entropy fails on the tail

Standard cross-entropy weights every example equally, so the gradient signal is dominated by the head classes;
the decision boundary is pushed to favor them, and rare-class recall collapses while overall accuracy stays
high. Worse, the model's *scores* become biased toward the priors: `p(y|x) ∝ p(x|y) p(y)`, so a rare class with
small prior `p(y)` is systematically under-predicted. Any honest treatment must (a) counter the prior bias
during training or inference and (b) report per-class metrics, because a single accuracy number is designed to
hide the tail.

## Remedies, from cheapest to most surgical

- **Class-weighted loss.** Weight each class inversely to frequency in `CrossEntropyLoss(weight=w)`, so rare
  classes contribute more gradient. Simple and often enough; over-weighting can destabilize training and hurt
  head-class recall.
- **Resampling.** Over-sample the minority (via `WeightedRandomSampler`) or under-sample the majority. Over-
  sampling risks over-fitting the few rare images; pair it with strong augmentation. Effective-number
  reweighting (Cui et al., 2019) uses `(1−β)/(1−β^{n_c})` instead of `1/n_c` to avoid over-counting near-
  duplicate rare samples.
- **Focal loss** (Lin et al., 2017, "Focal Loss for Dense Object Detection"). Down-weight easy examples so
  training focuses on the hard, usually-rare ones:

      FL(p_t) = − (1 − p_t)^γ log(p_t),

  where `p_t` is the model's probability of the true class and `γ ≥ 0` is the focusing parameter. When the
  model is already confident and correct (`p_t → 1`), the modulating factor `(1−p_t)^γ → 0` and that example
  contributes almost nothing; hard examples (`p_t` small) keep near-full weight. `γ = 2` is a common default;
  `γ = 0` recovers cross-entropy. Focal loss was designed for the extreme foreground/background imbalance of
  dense object detection but transfers to any long-tailed classification.
- **Logit adjustment / decoupling.** Menon et al. (2021, "Long-tail learning via logit adjustment") add
  `log p(y)` to the logits so the decision is corrected for the class prior in a Bayes-consistent way. Kang et
  al. (2020, "Decoupling Representation and Classifier") show a strong recipe: train the *feature extractor* on
  the natural (imbalanced) distribution, then *re-balance only the classifier head* — representations and the
  decision boundary want different sampling.

## Choosing by cost, not by leaderboard

There is no universally best remedy; the right one depends on the **cost of each error**. A rare-disease screen
wants high tail recall even at the price of head-class precision — weight the loss or adjust logits toward the
tail. A product-catalog tagger where a missed rare tag is cheap may prefer to leave the head untouched. Always
report the trade: per-class recall *before and after* the remedy, so the reader sees what improving the tail
cost the head. Optimizing "balanced accuracy" or macro-F1 rather than raw accuracy makes the objective match the
intent.

## Reporting honestly under distribution shift

The deepest issue is that your test set may not be your deployment distribution — the tail proportions, the
cameras, the demographics can all shift. Two disciplines keep you honest. First, **report the operating
assumptions**: state the test-set class distribution, and if deployment differs, re-weight metrics to the
expected distribution (importance weighting) rather than quoting a number computed under the wrong prior.
Second, **subgroup evaluation**: aggregate accuracy can hide large disparities across subgroups. Buolamwini &
Gebru (2018, "Gender Shades") found commercial face classifiers with near-perfect aggregate accuracy but error
rates up to 34% higher on darker-skinned women than lighter-skinned men — a failure visible only in
disaggregated, subgroup-level metrics. For any classifier that touches people, report per-subgroup performance,
not just the average, and treat a large disparity as a defect, not a footnote.

## Calibration meets the tail

Recall Lecture 2: networks are over-confident, and re-balancing can make it worse (up-weighting rare classes
inflates their scores). After applying an imbalance remedy, re-check the reliability diagram and ECE, and
temperature-scale on a *balanced* or distribution-matched validation split so the calibration target matches
deployment. A model that is both accurate on the tail and calibrated is far more useful downstream than one that
merely tops a single accuracy number.

## Common pitfalls

- **Reporting only overall accuracy on imbalanced data.** It is engineered to hide the tail; always disaggregate
  by class and by subgroup.
- **Over-sampling without augmentation.** Duplicating a handful of rare images just memorizes them; augment
  aggressively when you resample.
- **Ignoring the prior at inference.** If deployment class frequencies differ from training, adjust logits or
  re-weight — do not quote a metric computed under the wrong distribution.

**Takeaway:** real data is long-tailed, and naive cross-entropy optimizes overall accuracy by abandoning the
tail. Counter it with class weighting, effective-number reweighting, resampling, focal loss, or logit
adjustment — chosen by the true cost of each error — and *always* report per-class and per-subgroup metrics with
their trade-offs. Honest reporting under imbalance and shift is not an ethics add-on; it is the difference
between a number and the truth.
