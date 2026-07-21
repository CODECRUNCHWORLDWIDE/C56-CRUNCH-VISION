# Week 1 — Images, pixels, color & filtering

> **Goal:** by Sunday you can load any image into a NumPy array or tensor, explain pixels, channels, and color spaces precisely, and implement 2-D convolution by hand to blur, sharpen, and find gradients — no library hiding the operation.

Welcome to **C56 · Crunch Vision**. Before a single neural network, we fix what every later week stands on: the **image as data**. An image is not a picture — it is a grid of numbers, and everything vision does is arithmetic on that grid. This week you learn what a pixel really is, how color is encoded, and the one operation — **convolution** — that runs from classical filtering all the way through to the CNNs of Week 3. You implement it by hand first, so when torchvision does it for you, it is never magic.

## Learning objectives

By the end of this week, you will be able to:

- **Load and inspect** images as arrays: understand shape, dtype, channel order, and value range.
- **Explain** color spaces — RGB, grayscale, HSV — and convert between them deliberately.
- **Implement** 2-D convolution by hand and use it to blur, sharpen, and detect gradients.
- **Distinguish** convolution from correlation and reason about kernels, padding, and stride.
- **Visualize** every step — histograms, channels, filtered outputs — because in vision you never trust a number you haven't seen.

## Prerequisites

- Python and NumPy basics.
- No prior computer-vision experience assumed.

## This week

**Lectures**

1. [Lecture 1 — An image is a grid of numbers](lecture-notes/01-what-is-an-image.md)
2. [Lecture 2 — Color spaces and channels](lecture-notes/02-color-spaces.md)
3. [Lecture 3 — Convolution: the one operation](lecture-notes/03-convolution-and-filtering.md)

**Exercises**

1. [Exercise 1 — Load, inspect, and visualize an image](exercises/exercise-01-load-and-inspect.md)
2. [Exercise 2 — Select an object by color in HSV](exercises/exercise-02-color-space-selection.md)
3. [Exercise 3 — Implement convolution and filter an image](exercises/exercise-03-convolution-by-hand.md)

**Challenges**

1. [Challenge 1 — Design a kernel for a purpose](challenges/challenge-01-kernel-design.md)
2. [Challenge 2 — Make convolution fast](challenges/challenge-02-separable-and-speed.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 2.

## Deliverable

A notebook that loads an image, converts between color spaces, and applies at least three hand-written convolution kernels (blur, sharpen, and an edge/gradient filter), showing each result next to the original.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
