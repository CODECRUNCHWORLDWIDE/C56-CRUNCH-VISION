# Exercise 1 — Load, inspect, and visualize an image

**Goal:** fluency with images as arrays — shape, dtype, range, and channels.

## Tasks

1. Load a color image with Pillow and convert it to a NumPy array. Print its `shape`, `dtype`, `min`, and `max`.
2. Split it into its R, G, and B channels and display each as a grayscale image. Notice which channel is brightest for a given colored object.
3. Convert the image to grayscale two ways: Pillow's `.convert("L")` and the weighted-sum formula `0.299R + 0.587G + 0.114B` by hand. Confirm they match closely.
4. Plot the histogram of pixel intensities for the grayscale image. Describe what a dark vs. bright image looks like in its histogram.

## Deliverable

A notebook showing the original, the three channels, both grayscale conversions, and the histogram — each labeled. The point is to *see* the numbers behind the picture.
