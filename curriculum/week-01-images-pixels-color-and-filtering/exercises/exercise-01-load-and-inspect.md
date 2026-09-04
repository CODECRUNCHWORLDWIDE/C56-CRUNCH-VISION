# Exercise 1 — Load, inspect, and interrogate the sensor pipeline

**Goal:** fluency with images as sampled, quantized, gamma-encoded arrays — shape, dtype, range,
channels, and the pipeline that produced them.

## Tasks

1. Load a color image with Pillow, convert to a NumPy array, and print `shape`, `dtype`, `min`, `max`. State
   the layout (`(H, W, C)`) and predict what a torchvision `(C, H, W)` version would look like.
2. Split into R, G, B channels and display each as grayscale. Note which channel is brightest for a given
   colored object and connect it to that object's color.
3. Convert to grayscale two ways: Pillow's `.convert("L")` and the hand-coded luma `0.299R + 0.587G +
   0.114B`. Confirm they match closely; explain any small difference (rounding, Rec.601 vs. Pillow's exact
   weights).
4. **Gamma:** treat the 8-bit values as sRGB-encoded. Linearize (`(v/255)**2.2`), then re-encode, and show
   round-trip is near-identity. Then blur the image once in gamma space and once in linear space (linearize →
   blur → re-encode) and show the linear-space blur preserves brightness on thin bright features.
5. Plot the intensity histogram of the grayscale image and describe how dark, bright, and low-contrast images
   differ in it. Apply histogram equalization and re-plot.

## Deliverable

A notebook showing the original, the three channels, both grayscale conversions, the gamma round-trip and the
gamma-space-vs-linear-space blur comparison, and the histogram before/after equalization — each labeled. The
point is to *see* the numbers, the color encoding, and the light behind the picture.
