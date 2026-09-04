# Week 7 — Reading List

Primary sources for Week 7. Start with the three starred essentials, then read for depth. Prefer these to blog posts — they are where the architectures, metrics, and losses are stated correctly.

- **Ronneberger, Fischer & Brox (2015), 'U-Net: Convolutional Networks for Biomedical Image Segmentation,' MICCAI** — the skip-connection encoder-decoder, designed for small-data segmentation; the canonical semantic-segmentation paper.
- **He, Gkioxari, Dollár & Girshick (2017), 'Mask R-CNN,' ICCV** — instance segmentation by adding an FCN mask branch to Faster R-CNN, with RoIAlign's sub-pixel bilinear sampling; read for detect-then-mask and why alignment matters.
- **Kirillov, He, Girshick, Rother & Dollár (2019), 'Panoptic Segmentation,' CVPR** — defines the unified task and the Panoptic Quality metric (PQ = SQ × RQ); the reference for how the two tasks were reconciled.
- Long, Shelhamer & Darrell (2015), 'Fully Convolutional Networks for Semantic Segmentation,' CVPR — the FCN that turned classifiers into dense predictors with learned upsampling and skips; the origin of the encoder-decoder for segmentation.
- Chen, Papandreou, Kokkinos, Murphy & Yuille (2018), 'DeepLab: Semantic Image Segmentation with Deep CNNs, Atrous Convolution, and Fully Connected CRFs,' TPAMI — atrous convolution and ASPP; the resolution-preserving alternative to encoder-decoders.
- Milletari, Navab & Ahmadi (2016), 'V-Net: ... Volumetric Medical Image Segmentation,' 3DV — introduces the soft-Dice loss for imbalanced targets; the source for Lecture 4's overlap-loss argument.
- Berman, Rannen Triki & Blaschko (2018), 'The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the IoU Measure,' CVPR — the differentiable Lovász extension of the Jaccard set function; the principled 'optimize the metric' loss.
- Lin, Goyal, Girshick, He & Dollár (2017), 'Focal Loss for Dense Object Detection,' ICCV — the (1−p_t)^γ modulation that down-weights easy pixels; widely reused in imbalanced segmentation.
- Cheng, Misra, Schwing, Kirillov & Girdhar (2022), 'Masked-attention Mask Transformer for Universal Image Segmentation (Mask2Former),' CVPR — one query-based architecture that is state of the art on semantic, instance, and panoptic; the unification of Lecture 5.
- Xie, Wang, Yu, Anandkumar, Alvarez & Luo (2021), 'SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers,' NeurIPS — efficient hierarchical-transformer encoder + all-MLP decoder; the practical transformer segmenter.
- Cordts, Omran, Ramos et al. (2016), 'The Cityscapes Dataset for Semantic Urban Scene Understanding,' CVPR — the benchmark, its fine-annotation cost (~1.5 h/image), and the driving-safety framing of per-class evaluation.
- Isensee, Jaeger, Kohl, Petersen & Maier-Hein (2021), 'nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation,' Nature Methods — evidence that a well-configured U-Net + Dice/CE beats fancier models; the governance/reproducibility lesson.
- Kirillov, Mintun, Ravi et al. (2023), 'Segment Anything (SAM),' ICCV — the promptable foundation segmenter trained on a billion masks; read for the foundation-model framing and its caveats.
