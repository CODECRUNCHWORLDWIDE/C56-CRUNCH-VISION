# Mini-Project — Transfer Learning on Your Own Small Dataset

## Brief

Build a real classifier on a *small* dataset the way professionals do — by standing on a pretrained backbone — and rigorously compare feature extraction against fine-tuning to decide what to ship.

## Requirements

1. **Data:** a small custom multi-class dataset (your own photos, or a small public one), with train/val/test splits and no leakage. Use the backbone's exact preprocessing.
2. **Backbone:** a pretrained torchvision model (ResNet, EfficientNet, MobileNet) with a swapped head.
3. **Two strategies:** (a) frozen feature extraction, and (b) fine-tuning the top block(s) with a small, ideally discriminative LR — both with augmentation and early stopping.
4. **Comparison:** accuracy, training time, and train/val gap for each strategy, plus per-class precision/recall and a confusion matrix for the better one.
5. **Recommendation:** a written decision — which strategy fits this dataset's size and domain, and why — referencing the four quadrants.
6. **README:** reproduce steps and an honest limitations section (including any pretraining-overlap risk).

## Stretch

- Add the tiny-data curve (accuracy vs. images-per-class) from Challenge 2.
- Try two different backbones and compare accuracy vs. model size (a Week-11 edge preview).

## What you're proving

You can build a strong classifier from little data using the mainstream method of applied vision, and *reason* about the transfer strategy rather than guessing. From here, the tasks get richer — next week you stop asking only "what is this image?" and start asking "what objects are in it, and where?".
