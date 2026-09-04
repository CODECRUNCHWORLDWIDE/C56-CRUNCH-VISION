# Mini-Project — The Capstone: a deployed, honestly-evaluated computer-vision system

## Brief

Design, build, train, evaluate, and deploy one computer-vision system end to end on a problem you choose —
at graduate rigor. This is the course, proven: not just a model that beats a baseline, but a model whose win
you can *defend statistically*, whose failures you have *measured and named*, and whose deployment you have
*scoped legally*.

## Requirements

1. **Framing.** A problem that matters, the *decision* it supports, the *simplest sufficient* task
   (classification/detection/segmentation), a dataset you are allowed to use (license + explicit
   privacy/consent/legal note; if it touches people, cite the applicable GDPR/BIPA/EU-AI-Act constraint and
   state intended vs. out-of-scope uses), a clear metric with its operating threshold, and a documented
   **baseline ladder** up to a zero-shot foundation model where feasible.
2. **Model.** An appropriate architecture (CNNs, transfer learning, detectors, segmenters, ViTs), trained
   with the training toolbox and transfer learning applied deliberately, with every design choice selected
   on *validation*, not test.
3. **Result with statistics.** Beat the *strongest* baseline on **held-out, group-split images** — or
   honestly analyze why you didn't — and *prove it*: report a confidence interval on your metric (Wilson for
   accuracy; group-aware bootstrap otherwise) and a paired significance test of the improvement (McNemar or
   paired bootstrap), stated as a one-sentence conclusion.
4. **Honest evaluation.** The task's right metric vs. baseline; a reliability diagram + ECE (with
   temperature scaling if miscalibrated); a proper scoring rule (NLL or Brier); a *visual* error gallery; a
   per-subgroup/dataset-bias audit reporting the worst-group metric and disparity; a robustness probe under
   a fixed corruption suite; and named failure modes.
5. **Deployment.** The model exported and served behind a `/predict` API with **golden-test-verified
   preprocessing parity**; if edge-targeted, optimized and benchmarked per Week 11. Reproducible setup
   (pinned deps, `train.py`/`eval.py`/serve entry points, saved artifacts) and a concrete
   **drift-monitoring plan**.
6. **Communication.** A README that leads with results-vs-baseline-*with-interval* and limitations; a
   complete **model card**; and a **datasheet** — together addressing privacy, consent, bias, and legal
   scope.

## Stretch

- Add the foundation-model robustness comparison (Challenge 3): does your fine-tuned model degrade faster or
  slower than a zero-shot model under corruption?
- Add a cost-sensitive operating-point analysis: pick the threshold that minimizes an explicit cost of
  false positives vs. false negatives for your decision, and justify it.

## What you're proving

Everything. Twelve weeks ago an image was a mystery grid of numbers; now you can take a computer-vision
problem from nothing to a *trustworthy, deployed, documented, and legally-scoped* system — and defend every
technical, statistical, and ethical choice in it. That's a computer-vision engineer. Where next?
[C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) to ship and monitor at scale, [C53 Crunch Nets](../C53-CRUNCH-NETS/)
for deeper architecture theory, and [C5 Crunch AI & Data Science](../C5-CRUNCH-AI-DATA-SCIENCE/) to widen
the foundation.

## Definition of done

Someone can clone your repo, follow the README, reproduce your headline result *with its confidence
interval*, send the API a real image and get a sensible, correctly-preprocessed prediction, and read an
honest account — statistics, calibration, per-subgroup fairness, robustness, failure modes, and legal
scope — of what the system does, where it fails, and for whom.
