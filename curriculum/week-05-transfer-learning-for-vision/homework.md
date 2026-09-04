# Week 5 — Homework

Make transfer learning second nature — it is how most real vision is built — and make the reasoning behind it explicit. These take a few focused hours and set up Week 6's move from 'what is this image?' to 'what objects are in it, and where?'. Do the derivations and the divergence measurement, not just the training run.

## Tasks

- Draw the four-quadrant grid (data size x domain similarity), write the recommended strategy in each cell, and add a note on where parameter-efficient fine-tuning (adapters/LoRA/BitFit) changes the small-data-different-domain cell.
- Explain, in writing, what catastrophic forgetting is and give four concrete ways to prevent it (small LR, warmup, head-first/LP-FT, layer-wise LR decay), tying each to why it protects pretrained weights.
- Write the Ben-David et al. (2010) bound from memory and, in one paragraph, map each of its three terms (source error, H-divergence, lambda) to a specific engineering lever you can pull.
- Add discriminative / layer-wise-decayed learning rates and warmup + cosine decay to your mini-project fine-tuning, and report the accuracy and train/val-gap change versus a single flat LR.
- Estimate the domain divergence for your target dataset (train a logistic domain classifier on frozen features) before and after fine-tuning, and report the change alongside target accuracy.
- Read the CLIP and DINO papers' transfer sections and the torchvision/`timm` model docs; list two backbones (one supervised, one self-supervised or vision-language) and their transfer trade-offs (frozen vs. fine-tuned, robustness, preprocessing).

## Definition of done

A committed project that adapts a pretrained backbone to a small custom multi-class dataset both as a frozen feature extractor/linear probe and via disciplined LP-FT fine-tuning (matched preprocessing, warmup + cosine + layer-wise-decayed LR, augmentation, early stopping), reporting a comparison of accuracy/time/overfitting, per-class metrics with a confusion matrix, a catastrophic-forgetting demonstration, an estimated domain-divergence reading placing the task on the four-quadrant grid, a pretraining-overlap/leakage audit, and a justified, theory-grounded recommendation of which strategy to ship.

Submit by committing your work to your course repo under `week-05/`.
