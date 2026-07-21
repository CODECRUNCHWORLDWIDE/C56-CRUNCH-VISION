# Mini-Project — An Object Detection System with Honest mAP

## Brief

Build a working object-detection system on images you care about, and evaluate it the way benchmarks do — with mAP — proving you understand boxes, IoU, NMS, and honest detection metrics.

## Requirements

1. **Detector:** run a COCO-pretrained detector, or fine-tune one on a small custom class (transfer learning).
2. **Core utilities:** your own `iou` and `nms`, cross-checked against torchvision, used in your pipeline.
3. **Inference:** draw boxes with class labels and confidence scores on a set of your own images at a justified confidence threshold.
4. **Ground truth:** hand-label (or reuse a labeled subset) enough images to evaluate.
5. **Evaluation:** per-class AP and mAP@0.5 (and mAP@0.75 if feasible), with true/false positive/negative counts.
6. **Error analysis:** name the failure modes — small objects, occlusion, class confusions — with example images, and note any ground-truth label issues.
7. **README:** reproduce steps, the metric definitions you used (which mAP), and honest limitations.

## Stretch

- Fine-tune on a custom class and report before/after mAP.
- Add the speed–accuracy frontier from Challenge 2.

## What you're proving

You can deploy and *honestly evaluate* an object detector — the most widely-used vision capability in industry. Next week you go finer still: not just a box around each object, but a pixel-perfect mask of its exact shape.
