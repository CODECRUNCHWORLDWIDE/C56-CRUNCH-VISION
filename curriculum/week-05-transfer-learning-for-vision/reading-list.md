# Week 5 — Reading List

Primary sources for Week 5. Start with the three starred essentials, then read for depth. Prefer these to blog posts — they are where transfer, adaptation, and modern pretraining are stated correctly.

- **Yosinski, Clune, Bengio & Lipson (2014), 'How transferable are features in deep neural networks?', NeurIPS** — the definitive layer-wise transferability experiment; specificity vs. co-adaptation, and why fine-tuning heals both.
- **Ben-David, Blitzer, Crammer, Kulesza, Pereira & Vaughan (2010), 'A theory of learning from different domains', Machine Learning** — the H-divergence bound: target error <= source error + divergence + adaptability; the theory under Lecture 4.
- **Kumar, Raghunathan, Jones, Ma & Liang (2022), 'Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution', ICLR** — why full fine-tuning can hurt OOD, and the LP-FT (probe-then-fine-tune) fix.
- Zeiler & Fergus (2014), 'Visualizing and Understanding Convolutional Networks', ECCV — feature visualizations showing near-universal early filters; the picture behind 'features are general'.
- Kornblith, Shlens & Le (2019), 'Do Better ImageNet Models Transfer Better?', CVPR — large study relating pretraining accuracy to downstream transfer, and where the correlation weakens.
- Raghu, Zhang, Kleinberg & Bengio (2019), 'Transfusion: Understanding Transfer Learning for Medical Imaging', NeurIPS — when ImageNet transfer helps little on distant (medical) domains; the exotic-domain caveat.
- Chen, Kornblith, Norouzi & Hinton (2020), 'A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)', ICML — the InfoNCE contrastive objective for self-supervised pretraining.
- Caron, Touvron, Misra, Jegou, Mairal, Bojanowski & Joulin (2021), 'Emerging Properties in Self-Supervised Vision Transformers (DINO)', ICCV — self-distillation SSL with strong frozen linear-probe features and emergent segmentation.
- He, Chen, Xie, Li, Dollar & Girshick (2022), 'Masked Autoencoders Are Scalable Vision Learners (MAE)', CVPR — masked image modeling; weak frozen, strong fine-tuned — the opposite tendency to DINO.
- Radford et al. (2021), 'Learning Transferable Visual Models From Natural Language Supervision (CLIP)', ICML — image-text contrastive pretraining enabling zero-shot classification and robust transfer.
- Wortsman et al. (2022), 'Robust fine-tuning of zero-shot models (WiSE-FT)', CVPR — weight-space interpolation between zero-shot and fine-tuned models to keep OOD robustness.
- Ganin et al. (2016), 'Domain-Adversarial Training of Neural Networks (DANN)', JMLR — driving the H-divergence toward zero by adversarial feature alignment; the bound turned into an algorithm.
