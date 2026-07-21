# Week 5 — Transfer learning for vision

> **Goal:** by Sunday you can take a pretrained backbone (ResNet, EfficientNet, or similar), adapt it to your own small dataset by feature extraction or fine-tuning, and reason about which strategy fits your data size and domain — the way nearly all real vision is actually done.

Almost nobody trains a vision model from scratch anymore. They start from a network already trained on millions of images (ImageNet) and **transfer** its learned features to their own task. This is the single highest-leverage technique in applied vision: it turns a 10-million-image problem into a 500-image one. This week you'll learn why transfer works, the spectrum from **feature extraction** to **full fine-tuning**, and how to pick the right point on that spectrum for *your* data.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** why features learned on ImageNet transfer to new tasks, and where the transfer breaks down.
- **Load** a pretrained backbone and adapt its classifier head to a new number of classes.
- **Apply** feature extraction (freeze the backbone) and full fine-tuning (train it all) and compare them.
- **Choose** a transfer strategy from the data-size × domain-similarity quadrants.
- **Fine-tune** responsibly — small learning rates, discriminative LRs, and honest evaluation.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — Why transfer learning works](lecture-notes/01-why-transfer-works.md)
2. [Lecture 2 — Feature extraction vs. fine-tuning](lecture-notes/02-feature-extraction-vs-finetuning.md)
3. [Lecture 3 — Fine-tuning responsibly](lecture-notes/03-finetuning-in-practice.md)

**Exercises**

1. [Exercise 1 — Load a backbone and adapt its head](exercises/exercise-01-load-and-swap-head.md)
2. [Exercise 2 — Train a frozen feature extractor](exercises/exercise-02-feature-extraction.md)
3. [Exercise 3 — Fine-tune and compare strategies](exercises/exercise-03-finetune-and-compare.md)

**Challenges**

1. [Challenge 1 — Push transfer to a different domain](challenges/challenge-01-domain-shift.md)
2. [Challenge 2 — How little data can you win with?](challenges/challenge-02-tiny-data.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 6.

## Deliverable

A pretrained backbone adapted to a small custom dataset two ways — as a frozen feature extractor and with fine-tuning — with a comparison of accuracy, training time, and overfitting, plus a written recommendation of which to ship and why.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
