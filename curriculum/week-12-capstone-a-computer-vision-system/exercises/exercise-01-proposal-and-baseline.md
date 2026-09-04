# Exercise 1 — Proposal, baseline ladder, and legal scoping

**Goal:** lock scope and legality before building.

## Tasks

1. Write a one-page proposal: the *decision* the system supports, the problem and why it matters, the task
   (classification/detection/segmentation) and *why the simplest sufficient one*, the dataset (with its
   license, and an explicit privacy/consent/legal note — if it touches faces or people, cite the applicable
   GDPR/BIPA/EU-AI-Act constraint and state your intended and out-of-scope uses), the exact metric and
   operating threshold, and the baseline you will beat.
2. Build a **baseline ladder**, not one baseline: (a) a trivial predictor, (b) a fine-tuned small model,
   and (c) a **zero-shot foundation model** (CLIP for classification, SAM for segmentation) where feasible.
   Report each on group-split held-out data.
3. Confirm that beating — or meaningfully comparing to — the *strongest* rung is realistic within the week.

## Deliverable

The proposal (including the legal-scoping paragraph) plus the baseline-ladder table with each metric on
held-out data and the `n` behind it. If you cannot get a baseline running, your scope is wrong — fix it on
day 1, not day 6.
