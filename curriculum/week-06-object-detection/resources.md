# Week 6 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- [PyTorch — TorchVision Object Detection Finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) — Faster R-CNN / Mask R-CNN fine-tuning end to end; the practical backbone for the mini-project.
- [torchmetrics — Mean Average Precision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) — a correct, tested COCO-mAP implementation to cross-check your own metric code against.
- [COCO dataset & evaluation documentation](https://cocodataset.org/#detection-eval) — the authoritative definition of mAP@[.5:.95], AP_S/M/L, and AR; read before quoting any mAP number.
- [Ultralytics YOLO documentation](https://docs.ultralytics.com/) — practical one-stage detection: training, inference, and export for the speed-accuracy challenge and Week-11 edge preview.
- Ren et al. (2015) *Faster R-CNN*, Redmon et al. (2016) *YOLO*, and Lin et al. (2017) *Focal Loss / RetinaNet* — the foundational architecture and loss papers; read the method sections, not just the abstracts.
- Carion et al. (2020) *DETR* — the set-prediction paper; read the Hungarian-matching loss section closely for Lecture 5 and Challenge 3.
- Jonathan Hui — *mAP (mean Average Precision) for Object Detection* (blog) — the clearest step-by-step walkthrough of the PR-curve-to-AP computation, useful alongside the COCO docs.
- Rezatofighi et al. (2019) *GIoU* and Zheng et al. (2020) *DIoU/CIoU* — the IoU-based regression losses; read for why the metric and the loss diverge and how the family repairs it.

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
