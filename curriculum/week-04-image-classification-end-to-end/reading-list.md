# Week 4 — Reading List

Primary sources for Week 4. Start with the three starred essentials, then read for depth. These are where the ideas — leakage, calibration, optimizer behavior, and long-tailed learning — are stated correctly.

- **Guo, Pleiss, Sun & Weinberger (2017), 'On Calibration of Modern Neural Networks,' ICML** — shows deep nets are systematically over-confident and that temperature scaling fixes it; the reference for Lecture 2's calibration section.
- **Lin, Goyal, Girshick, He & Dollár (2017), 'Focal Loss for Dense Object Detection' (RetinaNet), ICCV** — introduces focal loss for extreme class imbalance; the core of Lecture 5.
- **Santurkar, Tsipras, Ilyas & Mądry (2018), 'How Does Batch Normalization Help Optimization?', NeurIPS** — debunks the 'internal covariate shift' story and shows BN smooths the loss landscape; essential for Lecture 4.
- Ioffe & Szegedy (2015), 'Batch Normalization,' ICML — the original method; read alongside Santurkar et al. for the corrected understanding.
- Loshchilov & Hutter (2019), 'Decoupled Weight Decay Regularization' (AdamW), ICLR — why Adam + L2 is wrong and how decoupling fixes it.
- Loshchilov & Hutter (2017), 'SGDR: Stochastic Gradient Descent with Warm Restarts,' ICLR — cosine annealing schedules used throughout the week.
- Smith (2017), 'Cyclical Learning Rates for Training Neural Networks,' WACV — the LR-range test for finding a good learning rate.
- Recht, Roelofs, Schmidt & Shankar (2019), 'Do ImageNet Classifiers Generalize to ImageNet?', ICML — fresh-test-set experiments quantifying how estimation bias inflates benchmark accuracy.
- Cui, Jia, Lin, Song & Belongie (2019), 'Class-Balanced Loss Based on Effective Number of Samples,' CVPR — the effective-number reweighting used in Exercise 4.
- Kang, Xie, Rohrbach, Yan, Gordo, Feng & Kalantidis (2020), 'Decoupling Representation and Classifier for Long-Tailed Recognition,' ICLR — train features on the natural distribution, rebalance only the head.
- Buolamwini & Gebru (2018), 'Gender Shades,' FAccT — subgroup disaggregation reveals disparities hidden by aggregate accuracy; the basis for Lecture 5's honest-reporting section.
- Karpathy (2019), 'A Recipe for Training Neural Networks' (blog/notes) — the canonical debugging discipline, including the overfit-a-tiny-batch sanity check.
