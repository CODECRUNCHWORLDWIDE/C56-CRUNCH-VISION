# Exercise 1 — Run a semantic segmenter and read the label field

**Goal:** produce, colorize, and critically read a per-pixel label map — and compare two architectural
families on the same images.

## Tasks

1. Load a pretrained semantic segmenter from torchvision (`deeplabv3_resnet50` or `fcn_resnet50`) with its exact
   preprocessing transform. Confirm you apply the *same* normalization the model was trained with — a common silent bug
   is skipping it, which quietly wrecks predictions.
2. Run it on 5–8 of your own images spanning easy (clear single object) to hard (clutter, unusual objects, tricky
   lighting). The raw output is `(1, num_classes, H, W)` logits; take `argmax` over the class dimension to get the label
   map.
3. Colorize the label map with a fixed class→color palette and overlay it semi-transparently (alpha≈0.5) on the
   original. Show original and overlay side by side.
4. **Second architecture.** Also run a transformer-based semantic segmenter (SegFormer via `transformers`, or a second
   torchvision model) on the *same* images and place the overlays next to the first model's. Note concretely where they
   agree and disagree — especially at boundaries and on small/unusual objects.
5. Identify, per image, at least one class the model handles well and one region it gets wrong; hypothesize the cause
   (boundary ambiguity, out-of-distribution object, scale).

## Deliverable

A notebook showing each image with colored semantic overlays from **two** models side by side, plus a short written
comparison of where boundaries are sharp vs. blurry and where the two architectures diverge. Looking at the masks —
and comparing families — is the point.
