# Week 12 — Quiz

Fifteen questions spanning statistical evaluation (intervals, paired tests, proper scoring), calibration, fairness auditing, deployment parity and drift, foundation-model baselines, and the law. Attempt each before the answer key.

**1. You report 88% accuracy on a 500-image test set. The single most important thing to add before calling this a result is:**

- A. the number of epochs
- B. a confidence interval (e.g., Wilson) and the n behind the estimate
- C. a second decimal place of precision
- D. the GPU model you trained on

<details>
<summary>Answer</summary>

**B. a confidence interval (e.g., Wilson) and the n behind the estimate** — A metric is a random variable; without an interval and the n, 0.88 is a point with unknown precision and cannot be compared honestly.

</details>

**2. To prove your model's accuracy genuinely beats the baseline on the same test set, the appropriate test is:**

- A. McNemar's test on the disagreement counts b and c
- B. reporting whichever model won on more random seeds
- C. a chi-square test on the confusion matrix
- D. an unpaired t-test on the two accuracies

<details>
<summary>Answer</summary>

**A. McNemar's test on the disagreement counts b and c** — Same-test-set comparison is paired; McNemar uses only the discordant pairs (b, c) and tests them against Binomial(b+c, 0.5).

</details>

**3. For a metric with no closed-form variance (mAP, mIoU, F1), the right way to get a 95% interval is:**

- A. divide the metric by the number of classes
- B. the Wald formula p̂ ± 1.96√(p̂(1−p̂)/n)
- C. assume it is exactly the clean number
- D. a nonparametric bootstrap: resample the test set with replacement B times and take percentiles

<details>
<summary>Answer</summary>

**D. a nonparametric bootstrap: resample the test set with replacement B times and take percentiles** — The bootstrap resamples the empirical distribution to approximate the sampling distribution of any metric, including ones with no closed form.

</details>

**4. Your test images are frames sampled from a few videos. When bootstrapping, you must resample:**

- A. individual frames, for more resamples
- B. pixels within each frame
- C. only the misclassified frames
- D. whole videos (groups), not individual frames

<details>
<summary>Answer</summary>

**D. whole videos (groups), not individual frames** — Correlated frames within a video are not independent; resampling frames understates variance exactly as within-video leakage does. Resample groups.

</details>

**5. A softmax output of 0.99 on a modern deep classifier means:**

- A. the model is right 99% of the time at that confidence
- B. nothing about accuracy, because modern nets are systematically overconfident
- C. the model is calibrated by construction
- D. the class is one-hot in training

<details>
<summary>Answer</summary>

**B. nothing about accuracy, because modern nets are systematically overconfident** — Guo et al. (2017) showed modern nets are overconfident; confidence must be validated with a reliability diagram/ECE before it is trusted.

</details>

**6. Expected Calibration Error (ECE) measures:**

- A. the number of misclassified images
- B. the variance of the logits
- C. the accuracy on the test set
- D. the accuracy-weighted average gap between confidence and empirical accuracy across bins

<details>
<summary>Answer</summary>

**D. the accuracy-weighted average gap between confidence and empirical accuracy across bins** — ECE = Σ (n_b/n)|acc(b) − conf(b)|; it summarizes how far confidence departs from realized accuracy across confidence bins.

</details>

**7. Temperature scaling to fix calibration works by dividing the logits by a scalar T, which:**

- A. changes the argmax and so improves accuracy
- B. leaves the argmax (accuracy) unchanged while adjusting the probabilities
- C. requires retraining the whole network
- D. only works for detection

<details>
<summary>Answer</summary>

**B. leaves the argmax (accuracy) unchanged while adjusting the probabilities** — Dividing all logits by one T is monotonic, so it cannot change the top class or accuracy; it only softens/sharpens the probability vector.

</details>

**8. A system that is 95% accurate overall but 68% on one underrepresented subgroup should be reported by leading with:**

- A. the worst-group metric and the disparity between groups
- B. the 95% overall number, since it is the average
- C. nothing about subgroups; overall accuracy suffices
- D. only the subgroup it does best on

<details>
<summary>Answer</summary>

**A. the worst-group metric and the disparity between groups** — Overall accuracy hides subgroup harm (Buolamwini & Gebru, 2018); the worst-group metric and disparity are the ethically load-bearing numbers.

</details>

**9. The proper scoring rule that punishes a confident wrong prediction most severely is:**

- A. negative log-likelihood (cross-entropy)
- B. the number of parameters
- C. mean IoU
- D. top-1 accuracy

<details>
<summary>Answer</summary>

**A. negative log-likelihood (cross-entropy)** — NLL sends the loss toward infinity as the probability assigned to the true class goes to zero, so a confident miss is penalized enormously.

</details>

**10. The single most common silent bug when deploying a vision model behind an API is:**

- A. the number of layers changed
- B. the GPU brand differs from training
- C. preprocessing mismatch: resize/normalize/channel-order differs from training
- D. the optimizer name is wrong

<details>
<summary>Answer</summary>

**B. the GPU brand differs from training** — Preprocessing parity failures throw no exception and quietly collapse accuracy; verify with a golden test comparing training-time and served outputs.

</details>

**11. In production you cannot compute accuracy because there are no labels. To monitor for drift you instead track:**

- A. proxies: predicted-class distribution, mean confidence, embedding-distribution distance, and spot checks
- B. the number of API calls only
- C. the model file size
- D. the training loss

<details>
<summary>Answer</summary>

**A. proxies: predicted-class distribution, mean confidence, embedding-distribution distance, and spot checks** — Without labels you monitor input/output distribution proxies and a small human-reviewed stream; shifts in these signal drift and can trigger a retrain.

</details>

**12. You tried 20 architectures and report the best one's test accuracy. Your reported number is:**

- A. unbiased, because the test set was held out
- B. biased upward, because you implicitly ran 20 tests and kept the luckiest
- C. biased downward
- D. unaffected by the number of models tried

<details>
<summary>Answer</summary>

**B. biased upward, because you implicitly ran 20 tests and kept the luckiest** — Selecting the max over many test-set evaluations is a multiple-comparisons problem; select on validation and evaluate the chosen model on test once.

</details>

**13. Why must your capstone compare against a zero-shot foundation model (e.g., CLIP or SAM)?**

- A. it is required by the grading rubric only
- B. foundation models cannot do zero-shot
- C. it guarantees your model will win
- D. it is often a very strong baseline; if your fine-tuned model can't beat it, that is the honest result

<details>
<summary>Answer</summary>

**D. it is often a very strong baseline; if your fine-tuned model can't beat it, that is the honest result** — In 2020s vision, zero-shot foundation models are frequently formidable baselines; beating one — or conceding it suffices — is a real, defensible finding.

</details>

**14. A capstone whose model classifies faces must, in its model card, address:**

- A. only the accuracy
- B. biometric-law constraints (e.g., GDPR Art. 9, BIPA consent) and intended vs. out-of-scope uses
- C. the learning rate schedule
- D. the GPU temperature

<details>
<summary>Answer</summary>

**B. biometric-law constraints (e.g., GDPR Art. 9, BIPA consent) and intended vs. out-of-scope uses** — Faces are biometric identifiers; consent and provenance are legal requirements, and the card must scope out prohibited uses (cf. EU AI Act).

</details>

**15. Recht et al. (2019) rebuilt ImageNet's test set by the original protocol and found accuracy dropped 11–14 points. The lesson for your capstone is:**

- A. test sets never matter
- B. distribution shift does not affect deep models
- C. clean test accuracy is a lower bound on field performance
- D. clean test accuracy is an upper bound; field performance under shift is usually worse

<details>
<summary>Answer</summary>

**D. clean test accuracy is an upper bound; field performance under shift is usually worse** — Models overfit specific benchmarks; a new same-distribution test set already drops accuracy, so your clean number over-states real-world performance.

</details>

---
