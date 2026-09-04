# Exercise 3 — Instance & panoptic segmentation: separating objects

**Goal:** separate individual objects with masks, and see the semantic-vs-instance-vs-panoptic distinction
made visual.

## Tasks

1. Load a pretrained Mask R-CNN (`maskrcnn_resnet50_fpn`, COCO weights). Run it on images with **multiple objects of the
   same class** (several people, several cars). Filter detections by a confidence threshold and inspect what changes as
   you vary it.
2. Draw each *instance's* mask in a distinct color, plus its box, class label, and score — showing the model separates
   object #1 from object #2. Note that overlapping instances can claim the same pixel (instance masks may overlap), unlike
   a semantic map.
3. **Semantic vs. instance, side by side.** On a similar multi-object scene, place your Exercise 1 semantic output next
   to this instance output. Make the distinction concrete: semantic merges all cars into one "car" blob; instance keeps
   car #1, #2, #3 separate.
4. **Panoptic (stretch within the exercise).** Run a panoptic model (e.g. Mask2Former panoptic via `transformers`, or
   `detectron2`'s panoptic head) on one scene and show the gap-free non-overlapping partition — every pixel labeled,
   things instanced, stuff classed. Contrast with the instance output, where background pixels are unlabeled.
5. Catalog failures: merged instances (two touching people as one), split instances (one object as two), and missed
   small objects.

## Deliverable

A notebook showing per-instance colored masks on a multi-object scene, a semantic-vs-instance side-by-side that makes
the distinction visual, and (stretch) a panoptic partition. Include a short note on merged/split/missed instances and
which failure each threshold change trades off.
