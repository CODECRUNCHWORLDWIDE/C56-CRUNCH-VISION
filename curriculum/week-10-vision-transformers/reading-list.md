# Week 10 — Reading List

Primary sources for Week 10. Start with the three starred essentials, then read for depth. Prefer these papers to blog summaries — they are where the results are stated correctly, with the caveats intact.

- **Dosovitskiy et al. (2021), 'An Image Is Worth 16x16 Words,' ICLR** — the original ViT: the patch-token reframe and the central data-hunger finding. The paper this week is built on.
- **Vaswani et al. (2017), 'Attention Is All You Need,' NeurIPS** — the Transformer and scaled dot-product / multi-head attention; the sqrt(d) scale is justified here. Read Sections 3.1–3.2.
- **Liu et al. (2021), 'Swin Transformer,' ICCV (best paper)** — shifted-window attention, linear cost, and a hierarchical backbone for dense tasks; the template for practical vision Transformers.
- Touvron et al. (2021), 'Training data-efficient image transformers & distillation through attention' (DeiT), ICML — a competitive ViT on ImageNet-1k via augmentation and a distillation token; the data-efficiency counterpoint.
- Liu et al. (2022), 'A ConvNet for the 2020s' (ConvNeXt), CVPR — a modernized pure CNN that matches ViTs, showing how much of the gap was the training recipe. Essential for honest comparison.
- He et al. (2022), 'Masked Autoencoders Are Scalable Vision Learners' (MAE), CVPR — 75%-mask reconstruction, asymmetric encoder-decoder; the dominant SSL pretext for ViTs.
- Caron et al. (2021), 'Emerging Properties in Self-Supervised Vision Transformers' (DINO), ICCV — self-distillation with no labels and the emergent object-segmenting attention maps.
- Radford et al. (2021), 'Learning Transferable Visual Models From Natural Language Supervision' (CLIP), ICML — contrastive image-text pretraining and zero-shot classification; the multimodal bridge.
- Dao et al. (2022), 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,' NeurIPS — the IO-aware tiling that makes exact attention practical at scale.
- Abnar & Zuidema (2020), 'Quantifying Attention Flow in Transformers,' ACL — attention rollout; the standard way to visualize what a ViT attends to across layers.
- Darcet et al. (2024), 'Vision Transformers Need Registers,' ICLR — high-norm artifact tokens in ViT attention and the register-token fix; a lesson in reading attention critically.
- Zhai et al. (2022), 'Scaling Vision Transformers,' CVPR — the compute/data/parameter scaling laws that make ViTs the frontier substrate.
