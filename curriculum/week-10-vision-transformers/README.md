# Week 10 — Vision Transformers

> **Goal:** by Sunday you can explain how a Vision Transformer turns an image into patch tokens and applies self-attention, implement patch embedding, run a pretrained ViT, and reason honestly about when ViTs beat CNNs and when they don't.

For a decade, convolution *was* vision. Then the Transformer — which had already conquered language — arrived for images. A **Vision Transformer (ViT)** cuts an image into patches, treats them like words, and lets **self-attention** relate every patch to every other. This week you'll understand the architecture reshaping the field: patch embedding, attention over patches, and the honest trade-offs. ViTs are data-hungry and powerful; knowing *when* they win over CNNs — and when they don't — is the mark of a current practitioner. Your [C53 Crunch Nets](../C53-CRUNCH-NETS/) Transformer knowledge carries directly over.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** how an image becomes a sequence of patch tokens with positional embeddings.
- **Describe** self-attention and why it gives a global receptive field from the first layer.
- **Implement** patch embedding and reason about attention's quadratic cost.
- **Run** a pretrained ViT and compare it to a CNN on the same task.
- **Judge** honestly when ViTs beat CNNs (data scale, transfer) and when CNNs remain the better choice.

## Prerequisites

- The previous week's mini-project, committed and working.

## This week

**Lectures**

1. [Lecture 1 — An image as a sequence of patches](lecture-notes/01-image-as-patches.md)
2. [Lecture 2 — Self-attention over patches](lecture-notes/02-self-attention-for-vision.md)
3. [Lecture 3 — ViT vs. CNN: an honest comparison](lecture-notes/03-vit-vs-cnn.md)

**Exercises**

1. [Exercise 1 — Implement patch embedding](exercises/exercise-01-patch-embedding.md)
2. [Exercise 2 — Run a pretrained ViT and inspect attention](exercises/exercise-02-run-a-vit.md)
3. [Exercise 3 — ViT vs. CNN on small data](exercises/exercise-03-vit-vs-cnn-small-data.md)

**Challenges**

1. [Challenge 1 — Global attention vs. local convolution](challenges/challenge-01-attention-vs-receptive-field.md)
2. [Challenge 2 — Interrogate the hype](challenges/challenge-02-efficiency-and-hype.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 11.

## Deliverable

A notebook implementing patch embedding (image → patch tokens) from scratch, running a pretrained ViT on your images, and comparing ViT vs. CNN predictions/accuracy on a small dataset with a written analysis of the trade-offs.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
