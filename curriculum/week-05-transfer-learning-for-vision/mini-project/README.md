# Mini-Project — Transfer Learning on Your Own Small Dataset, Done Properly

## Brief

Build a real classifier on a *small* dataset the way professionals do — by standing on a pretrained backbone —
and rigorously compare feature extraction against fine-tuning to decide what to ship, with the discipline and
the theory of this week behind every choice.

## Requirements

1. **Data.** A small custom multi-class dataset (your own photos, or a small public one), with clean
   train/val/test splits and *no leakage*. Use the backbone's **exact** preprocessing (`weights.transforms()`).
2. **Backbone.** A pretrained torchvision/`timm` model with a swapped head. State which backbone and why it
   fits your domain and compute.
3. **Two strategies.** (a) **Frozen feature extraction / linear probe** — cache features and probe them; and
   (b) **disciplined fine-tuning** of the top block(s) with a **small, warmed-up, cosine-decayed, layer-wise-
   decayed** LR, augmentation, weight decay, and early stopping. Use **LP-FT** ordering: probe first, then
   fine-tune.
4. **Comparison.** Accuracy, training time, and train/val gap for each strategy, plus per-class precision/recall
   and a confusion matrix for the better one. Include the deliberate **catastrophic-forgetting** demonstration
   (too-large LR) as a cautionary data point.
5. **Domain reading.** Estimate the domain divergence (Exercise 4 method: a domain classifier on features) and
   place your task on the data-size x domain-distance grid; state which shift regime (covariate vs. concept) you
   believe you are in and why.
6. **Recommendation.** A written decision — which strategy fits this dataset's size and domain, and why —
   referencing the four quadrants and, where relevant, the source-error / divergence / adaptability terms.
7. **README.** Reproduce steps and an honest limitations section, including a **pretraining-overlap / leakage
   audit** and any small-test-set noise (report a cross-validated estimate or interval where feasible).

## Stretch

- Add the **tiny-data curve** (accuracy vs. images-per-class) from Challenge 2, with a from-scratch baseline.
- Swap in a **self-supervised or CLIP backbone** (Lecture 5); compare frozen-vs-fine-tuned behaviour and, for
  CLIP, add the zero-shot point. Try **WiSE-FT** if you have an OOD set.
- Compare two backbones on accuracy vs. model size (a Week-11 edge preview).

## Definition of done — what you are proving

You can build a strong classifier from little data using the mainstream method of applied vision, fine-tune it
*responsibly*, evaluate it *honestly*, and — crucially — **reason** about the transfer strategy with theory
rather than guess. From here the tasks get richer: next week you stop asking only "what is this image?" and
start asking "what objects are in it, and where?".
