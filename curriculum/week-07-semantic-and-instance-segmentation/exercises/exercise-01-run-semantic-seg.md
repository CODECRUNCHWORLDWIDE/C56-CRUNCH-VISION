# Exercise 1 — Run a semantic segmenter

**Goal:** produce and read a per-pixel label map.

## Tasks

1. Load a pretrained semantic segmentation model from torchvision (e.g. `deeplabv3_resnet50` or `fcn_resnet50`) with its correct preprocessing.
2. Run it on several of your own images. The output is a per-pixel class map.
3. Colorize the class map (a color per class) and overlay it semi-transparently on the original image.
4. Identify classes the model handles well and pixels it gets wrong (especially boundaries and small/unusual objects).

## Deliverable

A notebook showing your images with colored semantic masks overlaid, and a short note on where the mask boundaries are accurate vs. blurry. Looking at the masks is the point.
