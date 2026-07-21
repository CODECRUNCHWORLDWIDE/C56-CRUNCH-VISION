# Week 12 — Quiz

Ten questions — a capstone-readiness check. Answer key below.

**1. The most common capstone failure is:**

- A. Too much documentation
- B. Too small a model
- C. Scoping too big/vague to finish or evaluate
- D. Using transfer learning

**2. When unsure which task to use, you should generally prefer:**

- A. The simplest task that solves your problem (classification over detection over segmentation)
- B. Whatever needs the most annotation
- C. None — skip framing
- D. The most complex (segmentation) always

**3. Before building a deep model you should establish:**

- A. A t-SNE plot
- B. A GPU cluster
- C. A metric and a baseline to beat
- D. A Vision Transformer

**4. Metrics must be reported on:**

- A. The training images
- B. Any images
- C. The validation set you tuned against
- D. Held-out images the model never saw and you didn't tune on

**5. To prevent leakage in a vision dataset, you split:**

- A. Randomly per image always
- B. By group — near-duplicates, same-subject photos, and same-video frames stay in one split
- C. By file size
- D. Not at all

**6. Vision's advantage for error analysis is that you can:**

- A. Only report one number
- B. Avoid held-out data
- C. Skip it
- D. See the mistakes — a gallery of misclassified images reveals patterns

**7. Reporting per-subgroup performance matters because:**

- A. It's faster
- B. Overall accuracy can hide poor performance on underrepresented groups — an ethical must
- C. It needs no data
- D. It replaces the baseline

**8. The #1 thing to verify when deploying a vision model is:**

- A. Preprocessing parity between training and the served pipeline
- B. The GPU brand
- C. The optimizer name
- D. The number of layers

**9. A model card for a vision system should include:**

- A. Intended use, data provenance, per-subgroup metrics, failure modes, and privacy/consent/bias notes
- B. The GPU temperature
- C. The learning rate only
- D. Only the accuracy

**10. The best capstones handle their weaknesses by:**

- A. Overclaiming results
- B. Hiding them
- C. Explaining them honestly — failures, bias, and next steps
- D. Ignoring evaluation

---

## Answer key

1. **C. Scoping too big/vague to finish or evaluate** — Scope for finished and evaluable, not maximal.
2. **A. The simplest task that solves your problem (classification over detection over segmentation)** — Annotation cost and difficulty rise sharply; use the simplest sufficient task.
3. **C. A metric and a baseline to beat** — Without a metric and baseline, success can't be judged.
4. **D. Held-out images the model never saw and you didn't tune on** — A peeked-at test set is just more training data.
5. **B. By group — near-duplicates, same-subject photos, and same-video frames stay in one split** — Correlated images across splits leak; group-splitting prevents it.
6. **D. See the mistakes — a gallery of misclassified images reveals patterns** — Visual error analysis exposes data, labeling, and real-limitation problems.
7. **B. Overall accuracy can hide poor performance on underrepresented groups — an ethical must** — Vision datasets are biased; a high overall number can mask harm to a subgroup.
8. **A. Preprocessing parity between training and the served pipeline** — Mismatched resize/normalize/channel-order silently collapses production accuracy.
9. **A. Intended use, data provenance, per-subgroup metrics, failure modes, and privacy/consent/bias notes** — It's how responsible ML is documented; vision's privacy weight makes it essential.
10. **C. Explaining them honestly — failures, bias, and next steps** — Honest limitations read as competence; overclaiming reads as the opposite.
