# Week 7 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- [PyTorch — semantic segmentation models](https://pytorch.org/vision/stable/models.html#semantic-segmentation) — DeepLabV3/FCN pretrained models with their exact preprocessing; the starting point for Exercise 1.
- [torchvision — Mask R-CNN / instance segmentation](https://pytorch.org/vision/stable/models.html#instance-segmentation-and-person-keypoint-detection) — pretrained `maskrcnn_resnet50_fpn` and its output format for Exercise 3.
- [Hugging Face `transformers` — SegFormer & Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) — ready-to-run transformer semantic/panoptic segmenters for the architecture comparison and Challenge 3.
- [torchmetrics — segmentation metrics](https://lightning.ai/docs/torchmetrics/stable/) — reference JaccardIndex/Dice/PQ implementations to validate your from-scratch metrics against.
- Jeremy Jordan — 'An overview of semantic image segmentation' — clear, correct explainer of architectures and the IoU/Dice metrics; a good first read alongside Lecture 2–3.
- *Dive into Deep Learning* (d2l.ai), semantic-segmentation and transposed-convolution chapters — free interactive treatment of FCN and learned upsampling.
- [COCO panoptic evaluation](https://cocodataset.org/#panoptic-eval) — the official PQ metric definition and evaluation code; pair with Lecture 3 and Challenge 3.
- Meta AI — Segment Anything (SAM) demo and repository — try promptable segmentation interactively to feel the foundation-model shift of Lecture 5 (use as an annotation aid, not a validated domain model).

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
