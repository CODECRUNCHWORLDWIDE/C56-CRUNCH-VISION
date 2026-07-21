# Week 7 — Homework

Cement dense prediction and its metrics before motion.

## Tasks

- Write one sentence each distinguishing semantic, instance, and panoptic segmentation.
- Explain why an encoder–decoder with skip connections beats a plain downsampling CNN for masks.
- Derive the relationship Dice = 2·IoU/(1+IoU) and state when you'd prefer Dice as a loss.
- Read the U-Net paper and the torchvision segmentation docs (in resources); note two architecture choices you'd reuse.

## Definition of done

A committed project that runs (or trains) a semantic or instance segmentation model on your images, overlays colored masks, and evaluates with mean IoU and/or Dice against ground-truth masks (never pixel accuracy alone), including per-class scores and a visual error analysis of boundary and small-object failures.

Submit by committing your work to your course repo under `week-07/`.
