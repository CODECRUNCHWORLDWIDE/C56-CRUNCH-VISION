# Lecture 2 — Metrics that tell the truth: proper scoring, calibration, and the regularization toolbox

A classifier's headline accuracy is the *least* informative number about it. Professional
evaluation means (a) choosing metrics that expose failure and carry their own uncertainty, (b) checking whether
the model's *probabilities* mean anything (calibration), and (c) regularizing deliberately — trading a little
training performance for a lot of generalization. This lecture makes each rigorous.

## Metrics beyond accuracy

- **Accuracy** — fraction correct. Fine for balanced data, misleading otherwise, and it discards the model's
  confidence entirely.
- **Per-class precision and recall** — precision = of images predicted class X, how many were X
  (`TP/(TP+FP)`); recall = of true class-X images, how many did we catch (`TP/(TP+FN)`). These expose *which*
  classes the model handles.
- **F1** — the harmonic mean `2·P·R/(P+R)`, useful when you need one number per class. **Macro-F1** averages
  per-class F1 equally (weights rare classes as much as common ones); **micro-F1** pools all TP/FP/FN first
  (weights by frequency). On imbalanced data they diverge sharply, and the choice is a values statement.
- **Confusion matrix** — the full picture of which classes get mistaken for which (cats↔dogs, huskies↔wolves).
  This single table drives most error analysis.
- **Top-k accuracy** — for many-class problems (ImageNet's 1000 classes), "is the true class in the top 5?" is
  the standard, fairer metric.

Pick metrics that match what you *care about*: a disease screen weights recall (never miss a case), a spam
filter weights precision (never flag good mail). The metric encodes the cost of each error.

## Accuracy is an estimate — give it error bars

`R̂_T` is a mean of `m` Bernoulli trials, so it has a standard error. A quick Wald interval is
`p̂ ± z·sqrt(p̂(1−p̂)/m)`; for small `m` or extreme `p̂` prefer the **Wilson** or **Clopper–Pearson** interval.
Concretely, 90% accuracy on `m = 200` test images has SE `≈ sqrt(0.9·0.1/200) ≈ 0.021`, so a 95% CI of roughly
`[86%, 94%]` — a 2-point difference between two models on this set is *noise*. Reporting accuracy without an
interval, or celebrating a sub-CI improvement, is a classic error. For paired model comparison on the *same*
test set, use **McNemar's test** on the discordant predictions rather than comparing two intervals.

## Proper scoring rules and calibration

Accuracy only cares whether `argmax` is right; it ignores the *probability* the model assigned. A model is
**calibrated** if, among predictions made with confidence `p`, a fraction `p` are actually correct. Deep
networks are notoriously **over-confident** — modern architectures often output 99% confidence at 80% accuracy
(Guo et al., 2017, "On Calibration of Modern Neural Networks"). Measure it with a **reliability diagram** (bin
predictions by confidence, plot accuracy per bin against confidence) and summarize with **Expected Calibration
Error**:

    ECE = Σ_b (|B_b|/N) · | acc(B_b) − conf(B_b) |,

the average gap between confidence and accuracy across confidence bins `B_b`. Cross-entropy (log loss) and the
Brier score are **proper scoring rules** — uniquely minimized by the true probabilities — so they reward
calibration, unlike accuracy. The cheap, effective fix is **temperature scaling**: after training, learn a
single scalar `T` on the validation set and replace logits `z` with `z/T` before softmax. `T > 1` softens
over-confident predictions; it changes no `argmax`, so accuracy is untouched while ECE often drops several-fold.
When your downstream decision uses the probability (triage thresholds, cost-sensitive routing, abstention),
calibration matters as much as accuracy.

## Regularization: the overfitting toolbox, as bias/variance

Overfitting — great on training, poor on held-out — is the default failure of a high-capacity model on finite
data. Each tool below buys lower variance, usually at a little bias:

- **Data augmentation** — usually the single most effective; it enlarges the effective training set with
  label-preserving transforms and encodes invariances (Week 3). More effective data beats every other trick.
- **Weight decay (L2)** — adds `(λ/2)‖θ‖²` to the loss, shrinking weights toward zero and biasing the model
  toward simpler functions. Set via the optimizer's `weight_decay`.
- **Dropout** (Srivastava et al., 2014) — randomly zeroes activations during training so no unit can be relied
  on; it approximates an ensemble over sub-networks and is strongest on the dense head.
- **Label smoothing** (Szegedy et al., 2016) — replace one-hot targets with `(1−ε)` on the true class and
  `ε/(K−1)` elsewhere; it discourages over-confident logits and often improves calibration and generalization.
- **Early stopping** — track validation loss and keep the checkpoint from *before* it started rising. Free, and
  it caps the effective capacity used.
- **Batch normalization** (Ioffe & Szegedy, 2015) — normalizes activations per mini-batch; stabilizes and
  accelerates training with a mild regularizing side effect (Lecture 4 explains the real mechanism).

```mermaid
flowchart LR
  A["Data augmentation"] --> B["Weight decay"]
  B --> C["Dropout"]
  C --> D["Label smoothing"]
  D --> E["Early stopping"]
```
*A rough order to reach for regularization, most effective first — each trades a little bias for less variance.*

The discipline: watch the **train-vs-validation gap**. Large gap → overfitting → add regularization or data.
Both losses high and close → underfitting → more capacity or longer training. You *diagnose* from the two
curves, then act.

## Common pitfalls

- **Comparing models by accuracy alone, within the confidence interval.** A 0.5-point win on 500 images is
  noise; test it, do not trust it.
- **Trusting the softmax probability as a probability.** Un-calibrated networks are over-confident; check ECE
  before using confidences for decisions.
- **Optimizing macro-F1 while reporting accuracy (or vice versa).** State which metric you optimize and why, in
  terms of error cost.

**Takeaway:** accuracy alone lies — report per-class precision/recall, a confusion matrix, top-k, *and* a
confidence interval, chosen to match what you care about. Check calibration with a reliability diagram and ECE,
and fix over-confidence with temperature scaling. Regularize with augmentation first, then weight decay,
dropout, and label smoothing, driving every choice from the train-vs-validation gap. Metrics are a values
statement; pick them on purpose.
