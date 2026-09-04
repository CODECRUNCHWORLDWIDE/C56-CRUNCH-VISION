# Week 12 — Reading List

Primary sources for the capstone — the statistics of trustworthy evaluation, calibration, fairness auditing, foundation-model baselines, and responsible-deployment documentation and law. Start with the three starred essentials; the rest are for depth and for citing in your model card.

- **Mitchell et al. (2019), 'Model Cards for Model Reporting,' ACM FAccT** — the standard structure for documenting a model's intended use, disaggregated metrics, and ethical considerations; your capstone's model card follows it directly.
- **Buolamwini & Gebru (2018), 'Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification,' PMLR/FAccT** — the canonical disaggregated fairness audit; the methodology behind your per-subgroup evaluation.
- **Guo, Pleiss, Sun & Weinberger (2017), 'On Calibration of Modern Neural Networks,' ICML** — shows modern nets are overconfident and introduces temperature scaling; the source for Lecture 2's calibration section.
- Dietterich (1998), 'Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms,' *Neural Computation* — why McNemar's test is the right paired comparison for classifiers on a shared test set.
- Efron & Tibshirani, *An Introduction to the Bootstrap* (1993), Ch. 6–13 — the rigorous treatment of bootstrap confidence intervals for arbitrary metrics; the basis of your mAP/mIoU/F1 intervals.
- Brown, Cai & DasGupta (2001), 'Interval Estimation for a Binomial Proportion,' *Statistical Science* — why the Wald interval fails and the Wilson/Agresti-Coull intervals are preferred for accuracy.
- Gneiting & Raftery (2007), 'Strictly Proper Scoring Rules, Prediction, and Estimation,' *JASA* — the theory of proper scoring rules (NLL, Brier) and the calibration/refinement decomposition.
- Gebru et al. (2021), 'Datasheets for Datasets,' *Communications of the ACM* — the standard for documenting dataset provenance, consent, and intended use; the basis of your capstone datasheet.
- Hendrycks & Dietterich (2019), 'Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,' ICLR (ImageNet-C) — the corruption-robustness protocol for your shift probe.
- Recht, Roelofs, Schmidt & Shankar (2019), 'Do ImageNet Classifiers Generalize to ImageNet?,' ICML — evidence that clean-benchmark accuracy overstates field performance; motivates treating your test number as an upper bound.
- Radford et al. (2021), 'Learning Transferable Visual Models From Natural Language Supervision (CLIP),' ICML — the zero-shot classification baseline your capstone must compare against.
- Kirillov et al. (2023), 'Segment Anything (SAM),' ICCV — the zero-shot promptable-segmentation baseline for segmentation capstones.
- European Parliament (2024), *Regulation on Artificial Intelligence (EU AI Act)*, Arts. 5 & Annex III — the prohibited and high-risk categories that constrain where a vision system may be deployed.
