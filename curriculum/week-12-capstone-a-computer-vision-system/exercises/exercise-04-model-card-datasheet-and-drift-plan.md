# Exercise 4 — Model card, datasheet, and drift-monitoring plan

**Goal:** document the system the way a review board and a production team both require.

## Tasks

1. Write a **model card** (Mitchell et al., 2019): intended use and out-of-scope uses, training data
   provenance and licenses, evaluation metrics *including per-subgroup and calibration*, known failure
   modes, robustness notes, and privacy/consent/legal considerations.
2. Write a one-page **datasheet** (Gebru et al., 2021) for your dataset: how it was collected, who/what is
   in it, what consent covers, known skews, and what it must not be used for.
3. Write a concrete **drift-monitoring plan**: which signals you would log in production (predicted-class
   distribution, mean confidence, an embedding-distribution distance, human-reviewed spot checks), what
   thresholds trip an alert, and what triggers a retrain. You cannot compute accuracy without labels in
   production — name the proxies you would use instead.

## Deliverable

`MODEL_CARD.md`, `DATASHEET.md`, and a `MONITORING.md` (or a monitoring section) committed to the repo. These
are graded artifacts, not afterthoughts — they are what make the system deployable by someone other than
you.
