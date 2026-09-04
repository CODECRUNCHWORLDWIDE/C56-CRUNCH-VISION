# Week 7 — Homework

Cement dense prediction, its metrics, and its loss choices before images start to move next week. Do the derivations by hand before coding — the IoU/Dice/PQ algebra and the imbalance argument must be instinct, not something a library hides.

## Tasks

- Write one crisp sentence each distinguishing semantic, instance, and panoptic segmentation *by their output space* (a class map, a set of masks, a gap-free labeled+instanced partition), not just by resolution.
- Derive Dice = 2·IoU/(1+IoU) from inclusion–exclusion, and state one scenario (small imbalanced target) where you would train on soft-Dice rather than cross-entropy, with the class-imbalance reason.
- Explain, with the receptive-field-vs-resolution tension, why an encoder-decoder with skip connections beats a plain downsampling CNN for masks — and how DeepLab's atrous convolution solves the same tension differently.
- Given a toy panoptic result (a few segments with stated IoUs, one FP and one FN), compute PQ, SQ, and RQ, and say in one sentence what SQ vs. RQ each tells you.
- Read the U-Net paper and either the Mask2Former or SegFormer paper; note two design choices you would reuse and one you would question, with reasons.
- For a high-stakes domain of your choice (medical, driving, or remote sensing), write a short paragraph naming the label-noise, calibration, and distribution-shift risks and how you would evaluate against each (per-class, out-of-distribution, reliability diagram).

## Definition of done

A committed project that runs (and ideally trains) a semantic and/or instance segmentation model on your own images, overlays colored masks, and evaluates with from-scratch, reference-validated per-class IoU and mIoU and/or Dice against ground-truth masks (never pixel accuracy alone) — including the small-object accuracy-trap demonstration, a stated loss rationale, a visual error gallery with named causes, and a README/model card that names the honest limitations (boundary-label ambiguity and, where relevant, calibration and distribution shift).

Submit by committing your work to your course repo under `week-07/`.
