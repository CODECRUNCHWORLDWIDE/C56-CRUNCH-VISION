# Week 6 — Reading List

Primary sources for Week 6 — the papers that defined the detection paradigms, the metric, and the loss. Start with the three starred essentials, then read for depth. Prefer these to blog summaries: they are where the ideas (and their exact conditions) are stated correctly.

- **Ren, He, Girshick & Sun (2015), 'Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,' NeurIPS** — the two-stage workhorse; read for the RPN, anchors, and shared-feature proposals.
- **Lin, Goyal, Girshick, He & Dollár (2017), 'Focal Loss for Dense Object Detection' (RetinaNet), ICCV** — the imbalance analysis and the focal-loss derivation that let one-stage match two-stage; the core of Lecture 4.
- **Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko (2020), 'End-to-End Object Detection with Transformers' (DETR), ECCV** — set prediction, Hungarian bipartite matching, and detection without anchors or NMS; the core of Lecture 5.
- Redmon, Divvala, Girshick & Farhadi (2016), 'You Only Look Once: Unified, Real-Time Object Detection,' CVPR — the original one-stage dense detector; read for the single-pass grid formulation.
- Liu, Anguelov, Erhan, Szegedy, Reed, Fu & Berg (2016), 'SSD: Single Shot MultiBox Detector,' ECCV — multi-scale one-stage detection and hard-negative mining.
- Lin, Dollár, Girshick, He, Hariharan & Belongie (2017), 'Feature Pyramid Networks for Object Detection,' CVPR — how multi-scale features handle the scale-variation problem; the backbone under most modern detectors.
- Tian, Shen, Chen & He (2019), 'FCOS: Fully Convolutional One-Stage Object Detection,' ICCV — the canonical anchor-free, center-based detector; read for the assignment rule.
- Rezatofighi, Tsoi, Gwak, Sadeghian, Reid & Savarese (2019), 'Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression,' CVPR — GIoU; why IoU fails as a loss and how to fix it.
- Bodla, Singh, Chellappa & Davis (2017), 'Soft-NMS — Improving Object Detection With One Line of Code,' ICCV — score-decay suppression for crowded scenes.
- Lin, Maire, Belongie, Hays, Perona, Ramanan, Dollár & Zitnick (2014), 'Microsoft COCO: Common Objects in Context,' ECCV — the dataset and, crucially, the exact mAP@[.5:.95] evaluation protocol you must quote precisely.
- Everingham, Van Gool, Williams, Winn & Zisserman (2010), 'The PASCAL Visual Object Classes (VOC) Challenge,' IJCV — the origin of the IoU-0.5 correctness convention and 11-point AP.
- Zhu, Su, Lu, Li, Wang & Dai (2021), 'Deformable DETR: Deformable Transformers for End-to-End Object Detection,' ICLR — the ~10× faster convergence and multi-scale fix to DETR (Lecture 5 successor).
