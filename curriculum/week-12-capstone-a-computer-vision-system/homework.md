# Week 12 — Homework

The capstone *is* the week; this homework is the graduate-rigor finishing that turns a model into a defensible system. Do the statistics and the calibration *before* you think you're done — they will change what you fix and what you claim. Every task here maps to a line in your defense.

## Tasks

- Compute a confidence interval on your capstone's headline metric — Wilson for accuracy or a group-aware bootstrap for mAP/mIoU/F1 — and put the interval (and n) in your README's results table, replacing any bare point estimate.
- Run the paired significance test of your model vs. baseline (McNemar with reported b, c, χ², p; or a paired-bootstrap interval of the difference), and write the one-sentence conclusion for your defense.
- Produce a reliability diagram and compute ECE for your model; if it is miscalibrated, fit temperature scaling on validation and report ECE before and after.
- Run a per-subgroup fairness audit, report the worst-group metric and the disparity, and write the intended/out-of-scope-use and privacy/consent/legal paragraph of your model card (cite GDPR/BIPA/EU-AI-Act if faces or people are involved).
- Build a small corruption suite, plot your metric vs. severity, and state which corruption breaks your model first and what that implies for deployment.
- Have someone else (or a fresh terminal) clone your repo, run eval.py to regenerate your numbers, and send the API a real image; fix whatever they can't reproduce and confirm the golden preprocessing-parity test passes.

## Definition of done

A committed, reproducible repository with: problem framing (decision + simplest sufficient task + metric + baseline ladder including a zero-shot foundation model); a trained vision model beating (or honestly analyzed against) its strongest baseline on held-out, group-split images *with a stated confidence interval and a paired significance test*; honest evaluation (task metric, reliability diagram + ECE, a proper scoring rule, visual error gallery, per-subgroup fairness audit with worst-group metric, corruption-robustness probe, named failure modes); a served /predict API with golden-test-verified preprocessing parity and a drift-monitoring plan; a model card + datasheet addressing privacy/consent/legal scope; a reproduce-me README; and a two-part written defense answering the hardest technical and ethical critique.

Submit by committing your work to your course repo under `week-12/`.
