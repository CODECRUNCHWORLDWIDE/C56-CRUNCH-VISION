# Challenge 1 — Fine-tune a detector on a custom class

Adapt a detector to find an object class it wasn't trained on — the applied core of detection.

1. Collect and label a small dataset (30–100 images) of a custom object with bounding boxes (use a tool like LabelImg or Roboflow's free tier, or draw boxes programmatically for a synthetic object).
2. Fine-tune a COCO-pretrained detector on your class (transfer learning from Week 5 applies — small LR, augmentation).
3. Evaluate mAP on a held-out split. Show detections on test images.
4. Do an error analysis: where does it fail — small instances, occlusion, unusual angles, backgrounds it confuses?

**Deliverable:** a fine-tuned detector, its held-out mAP, example detections, and a named-failure-mode analysis. Labeling your own data teaches you how much annotation quality and quantity drive detection — a lesson no tutorial dataset delivers.
