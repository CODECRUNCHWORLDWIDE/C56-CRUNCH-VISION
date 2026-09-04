# Exercise 2 — Run a pretrained detector and dissect its raw output

**Goal:** get real detections on your own images and understand what the model emits *before* the
convenience wrappers clean it up.

## Tasks

1. Load a COCO-pretrained detector from torchvision (e.g. `fasterrcnn_resnet50_fpn` or
   `retinanet_resnet50_fpn`) or a YOLO model. Print the number of classes and confirm the class-index → name
   mapping (COCO's 80 classes, with the off-by-one background-index gotcha in some APIs).
2. Run it on several of your own photos. Inspect the *raw* output dict: boxes, labels, scores, and how many
   detections come back before any confidence filtering.
3. Draw surviving boxes (filtered at a sensible confidence, e.g. 0.5) with class labels and scores on each
   image.
4. **Confidence sweep.** Show, on one image, how a very low threshold (0.05) floods it with false boxes and a
   very high one (0.9) misses real objects; plot number-of-detections vs. threshold. Confirm the model already
   applied NMS internally (look for the absence of obvious duplicates even at low confidence).
5. **Latency.** Time inference over 20 images (warm the model first; time only the forward pass) and report
   mean ms/image on your hardware — you will reuse this in Challenge 2.

## Deliverable

A notebook showing your images with drawn detections at a justified confidence threshold, the
detections-vs-threshold plot, the confirmed class mapping, and a mean-latency number. Note any obviously wrong
or missed detections — that is the seed of the error analysis in the mini-project.
