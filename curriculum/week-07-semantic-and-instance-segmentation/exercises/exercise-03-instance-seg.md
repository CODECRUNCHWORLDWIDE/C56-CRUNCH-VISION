# Exercise 3 — Run instance segmentation

**Goal:** separate individual objects with masks.

## Tasks

1. Load a pretrained Mask R-CNN (`maskrcnn_resnet50_fpn`) with COCO weights.
2. Run it on images with multiple objects of the same class (several people, several cars). Filter by confidence.
3. Draw each *instance's* mask in a distinct color, plus its box, label, and score — showing the model separates object #1 from object #2.
4. Contrast with your semantic-segmentation output from Exercise 1 on a similar scene: semantic merges same-class objects; instance separates them.

## Deliverable

A notebook showing per-instance colored masks on a multi-object scene, side by side with the semantic result, making the semantic-vs-instance distinction visual. Note any merged or missed instances.
