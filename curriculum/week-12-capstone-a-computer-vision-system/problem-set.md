# Week 12 — Graduate Problem Set: the statistics and ethics of a shipped vision system

Ten problems, easy to hard, mixing computation, derivation, proof, and open analysis. Solution
sketches are at the end — attempt each fully before reading them. Throughout, `n` is test-set size, `p̂` an
observed metric, `z = 1.96` the 95% normal quantile.

**P1 (interval, warm-up).** A classifier gets 456 of 600 test images right. Compute `p̂`, its standard error
`√(p̂(1−p̂)/n)`, and the 95% Wald interval. In one sentence, why might you prefer the Wilson interval here
if `p̂` were instead 0.995?

**P2 (sample size).** You want a 95% interval on accuracy of half-width at most ±0.02. Using the worst-case
variance (`p=0.5`), how many test images do you need? Recompute the requirement if you expect `p̂ ≈ 0.9`.

**P3 (McNemar computation).** On the same 600-image test set, your model and the baseline disagree on 90
images: your model is right and the baseline wrong on `b = 58`, and the reverse on `c = 32`. Compute the
continuity-corrected McNemar statistic `χ² = (|b−c|−1)²/(b+c)` and state whether the difference is
significant at α=0.05 (critical value 3.84). Why do the ~510 images they agree on not appear?

**P4 (why paired beats unpaired).** Argue, in terms of variance, why comparing two models on the *same* test
set with McNemar is more powerful than comparing two independently-measured accuracies with an unpaired
test. What source of variance does pairing remove?

**P5 (bootstrap procedure).** Write pseudocode for a paired bootstrap that produces a 95% interval on
`mAP_model − mAP_baseline` for a detection capstone whose test images come from 40 distinct scenes (several
images per scene). Identify the resampling unit and justify it.

**P6 (ECE by hand).** You have three confidence bins. Bin [0.6,0.8): 100 images, mean confidence 0.70,
accuracy 0.62. Bin [0.8,0.9): 200 images, mean confidence 0.85, accuracy 0.80. Bin [0.9,1.0]: 300 images,
mean confidence 0.96, accuracy 0.88. Compute the ECE. Is the model over- or under-confident, and where
worst?

**P7 (temperature scaling, proof).** Prove that scaling all logits `z_k → z_k / T` for a fixed `T > 0` never
changes `argmax_k z_k`, and hence never changes accuracy. Then explain why `T > 1` reduces confidence and
`T < 1` increases it.

**P8 (proper scoring rule).** Show that for a single binary example with true label `y ∈ {0,1}` and
predicted probability `q` for class 1, the expected cross-entropy loss under the true probability `p` is
minimized (over `q`) at `q = p`. (This is what "proper" means.) What does this imply about a model that
minimizes cross-entropy on infinite data?

**P9 (multiple comparisons).** You evaluate 20 candidate models on the test set and report the best
accuracy. Assuming all 20 truly have accuracy 0.85 with test-set standard error 0.015, roughly how much can
the *maximum* of 20 such estimates exceed 0.85 in expectation? State the fix (validation selection;
Bonferroni if you must compare finalists on test).

**P10 (open — legal and ethical scoping).** Your capstone is a face-attribute classifier trained on a
scraped web dataset. Identify at least three concrete problems (data provenance/consent, GDPR/BIPA exposure,
EU-AI-Act prohibited-use risk, subgroup bias) and, for each, state what you would change in the data, the
evaluation, or the model card's intended-use scope to make the system defensible — or argue it should not be
built at all. Argue carefully rather than list.

---

## Solution sketches

**S1.** `p̂ = 456/600 = 0.760`; SE `= √(0.76·0.24/600) = √(0.0003040) ≈ 0.0174`; Wald `0.760 ± 1.96·0.0174 =
0.760 ± 0.0341` → [0.726, 0.794]. At `p̂ = 0.995` the Wald interval would extend above 1.0 (nonsensical); the
Wilson interval stays inside [0,1] and is asymmetric, so it is preferred near the boundary and for small `n`.
**S2.** Worst case `p=0.5`: half-width `= z√(0.25/n) ≤ 0.02` ⇒ `n ≥ (z²·0.25)/0.02² = (3.8416·0.25)/0.0004 =
2401`. For `p̂≈0.9`: variance `0.09`, `n ≥ (3.8416·0.09)/0.0004 = 864` (about 865). Accuracy near the
boundary needs fewer samples for the same absolute width.
**S3.** `χ² = (|58−32|−1)²/(58+32) = (25)²/90 = 625/90 ≈ 6.94 > 3.84`, so significant at α=0.05 (p≈0.008). The
~510 concordant images (both right or both wrong) carry no information about *which* model is better — under
the null the informative quantity is how the *discordant* pairs split, tested against 50/50.
**S4.** Unpaired testing sees the full image-to-image difficulty variance in *both* accuracy estimates;
pairing on the same images subtracts out that shared difficulty, leaving only the model-vs-model difference
per image. Lower variance ⇒ a real difference is detectable with fewer images. Pairing removes the
*item-difficulty* variance.
**S5.** For b in 1..B: sample 40 scenes *with replacement*; take all images from each sampled scene; compute
`mAP_model` and `mAP_baseline` on that pooled set; store the difference. Report the 2.5/97.5 percentiles of
the B differences. Resampling unit = **scene**, because images within a scene are correlated; resampling
images would understate variance (same reasoning as group-splitting to prevent leakage).
**S6.** ECE `= Σ (n_b/n)|acc−conf|`. Weights: 100/600, 200/600, 300/600. Gaps: |0.62−0.70|=0.08,
|0.80−0.85|=0.05, |0.88−0.96|=0.08. ECE `= (1/6)(0.08)+(1/3)(0.05)+(1/2)(0.08) = 0.01333+0.01667+0.04 =
0.0700`. In every bin accuracy < confidence ⇒ **over-confident**; the largest weighted contribution is the
high-confidence bin (0.04), so overconfidence is worst exactly where the model is most sure — the dangerous
case.
**S7.** `argmax_k z_k/T = argmax_k z_k` because dividing by `T>0` is a strictly increasing transform that
preserves the order of the logits; the maximal index is unchanged, so accuracy is unchanged. `T>1` shrinks
the logit gaps ⇒ softmax outputs move toward uniform ⇒ lower confidence; `T<1` widens gaps ⇒ sharper,
higher confidence.
**S8.** Expected loss `= −[p log q + (1−p) log(1−q)]`. Differentiate w.r.t. `q`: `−[p/q − (1−p)/(1−q)] = 0`
⇒ `p(1−q) = (1−p)q` ⇒ `p = q`. Second derivative positive ⇒ minimum. So minimizing cross-entropy on infinite
data recovers the *true* conditional probabilities — the model becomes calibrated, which is why NLL is a
proper scoring rule.
**S9.** The expected maximum of 20 iid N(0.85, 0.015²) estimates exceeds the mean by roughly the expected
max of 20 standard normals (~1.87) times 0.015 ≈ 0.028, so the reported best ≈ 0.878 purely by selection —
an inflation of ~2.8 points with no real improvement. Fix: select the model on validation and evaluate the
single chosen model on test once; if you must compare `m` finalists on test, use Bonferroni (α/m) or report
that you did.
**S10.** Sketch: (1) *Provenance/consent* — scraped faces lack consent; GDPR Art. 9 treats biometric-ID data
as special-category needing an explicit lawful basis, BIPA requires written consent. Fix: use a
consented/licensed dataset or don't build it. (2) *EU AI Act* — untargeted facial scraping to build
recognition databases is prohibited; face-attribute inference can be high-risk. Fix: scope intended use to
exclude identification/surveillance and document out-of-scope uses. (3) *Subgroup bias* — attribute
classifiers historically fail on darker skin tones and non-binary presentations. Fix: disaggregated
evaluation with worst-group reporting, and refuse to ship if a subgroup is unacceptably worse. A defensible
version narrowly scopes use, uses consented data, audits by subgroup, and states prohibited uses; if none of
that is achievable, the correct answer is not to build it.
