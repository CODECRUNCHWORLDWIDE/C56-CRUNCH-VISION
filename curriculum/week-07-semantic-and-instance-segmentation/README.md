# Week 7 — Semantic & instance segmentation

> **Goal:** by Sunday you can explain the difference between semantic, instance, and panoptic segmentation, describe encoder–decoder and Mask R-CNN architectures, run a pretrained segmenter on your images, and evaluate masks with IoU/Dice and mean IoU.

Detection draws a box around each object. **Segmentation** goes finer — it labels *every pixel*. That precision matters enormously: a medical model outlining a tumor's exact shape, a self-driving car separating road from sidewalk pixel by pixel, a photo app cutting out a person for a background blur. This week covers **semantic** segmentation (every pixel gets a class), **instance** segmentation (separate each object), and **panoptic** (both), the encoder–decoder architectures that produce masks, and how to measure them honestly.

## Learning objectives

By the end of this week, you will be able to:

- **Distinguish** semantic, instance, and panoptic segmentation by what each labels.
- **Explain** the encoder–decoder (U-Net) architecture and why skip connections matter for masks.
- **Describe** how Mask R-CNN adds a mask branch to a detector for instance segmentation.
- **Run** a pretrained segmentation model and overlay its masks on your images.
- **Evaluate** segmentation with pixel IoU, Dice, and mean IoU — and know why pixel accuracy misleads.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Semantic, instance & panoptic segmentation](lecture-notes/01-kinds-of-segmentation.md)
2. [Lecture 2 — Encoder–decoders, U-Net & Mask R-CNN](lecture-notes/02-encoder-decoder-architectures.md)
3. [Lecture 3 — Measuring segmentation honestly](lecture-notes/03-segmentation-metrics.md)

**Exercises**

1. [Exercise 1 — Run a semantic segmenter](exercises/exercise-01-run-semantic-seg.md)
2. [Exercise 2 — Implement IoU and Dice for masks](exercises/exercise-02-iou-dice.md)
3. [Exercise 3 — Run instance segmentation](exercises/exercise-03-instance-seg.md)

**Challenges**

1. [Challenge 1 — Train a U-Net on a small dataset](challenges/challenge-01-train-a-unet.md)
2. [Challenge 2 — Stress-test a segmenter and choose a loss](challenges/challenge-02-segmentation-in-the-wild.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 8.

## Deliverable

A notebook that runs a pretrained semantic or instance segmentation model on your images, overlays colored masks, and computes IoU/Dice against at least a few hand-drawn ground-truth masks, with an honest note on where the masks are wrong.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
