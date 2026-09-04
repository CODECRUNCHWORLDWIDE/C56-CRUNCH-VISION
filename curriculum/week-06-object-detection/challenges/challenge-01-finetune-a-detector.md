# Challenge 1 — Fine-tune a detector on a custom class and analyze its failures

Adapt a detector to find an object class it was never trained on — the applied core of detection,
and the challenge that teaches how much annotation quality drives everything.

1. **Collect and label** a small dataset (30-100 images) of a custom object with bounding boxes (LabelImg,
   Roboflow's free tier, CVAT, or programmatically drawn boxes for a synthetic object). Split into
   train/val/test; keep the test split untouched until the end.
2. **Fine-tune** a COCO-pretrained detector on your class (transfer learning from Week 5: small learning rate,
   freeze or lightly tune the backbone, augment). Replace/extend the classification head for your class count.
3. **Evaluate** mAP@0.5 and mAP@0.75 on the held-out test split using *your* metric code from Exercise 3, and
   show example detections.
4. **Named-failure-mode error analysis.** Categorize every test-set miss and false positive into named modes:
   small instances, occlusion, unusual viewpoints, background confusions, and — importantly — **ground-truth
   label errors** in your own annotations. Quantify how many failures fall in each bucket and show one example
   image per mode.
5. **Data-scale probe (stretch).** Re-train on 25% / 50% / 100% of your labels and plot mAP vs. training-set
   size to see how far you are from the data plateau.

**Deliverable:** a fine-tuned detector, its held-out mAP@0.5 and 0.75 (computed with your own code), example
detections, and a *quantified* named-failure-mode analysis with one example per mode. Labeling your own data
teaches how much annotation quality and quantity drive detection — a lesson no tutorial dataset delivers.
