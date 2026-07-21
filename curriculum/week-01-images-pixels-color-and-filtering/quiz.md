# Week 1 — Quiz

Ten questions. Answers and reasoning at the bottom — try each before you scroll.

**1. A standard color image loaded from Pillow has array shape:**

- A. (H, W, 3)
- B. (H, W, 4) always
- C. (H, W)
- D. (3, H, W)

**2. An 8-bit grayscale pixel value of 0 represents:**

- A. Transparent
- B. Red
- C. White
- D. Black

**3. OpenCV loads color images in channel order:**

- A. RGB
- B. HSV
- C. BGR
- D. Grayscale

**4. Converting RGB to grayscale with a weighted sum weights green most because:**

- A. Green pixels are rarer
- B. It saves memory
- C. Green is the first channel
- D. Human vision is most sensitive to green/brightness

**5. HSV is useful for color selection because:**

- A. It uses less memory
- B. It removes all color
- C. Hue stays roughly constant under lighting changes while brightness moves to V
- D. It is faster to convolve

**6. A 3×3 convolution kernel of all 1/9 values performs a:**

- A. Box blur (average)
- B. Rotation
- C. Sharpen
- D. Edge detection

**7. For input size n, kernel k, padding p, stride s, the output size is:**

- A. floor((n + 2p − k)/s) + 1
- B. n/s only
- C. n × k
- D. n − k

**8. What most deep-learning libraries call 'convolution' is technically:**

- A. True convolution with a flip
- B. Matrix inversion
- C. Cross-correlation (no kernel flip)
- D. A Fourier transform

**9. A Sobel kernel is used to:**

- A. Convert color spaces
- B. Resize the image
- C. Detect intensity gradients (edges)
- D. Blur noise

**10. A Gaussian blur is 'separable', meaning it can be computed as:**

- A. A 1-D horizontal pass then a 1-D vertical pass
- B. It cannot be sped up
- C. One 2-D pass only
- D. A single matrix inverse

---

## Answer key

1. **A. (H, W, 3)** — Pillow gives height × width × 3 channels (RGB). PyTorch batches instead use (N, C, H, W).
2. **D. Black** — In 8-bit intensity, 0 is black and 255 is white.
3. **C. BGR** — OpenCV's historical convention is BGR, which mismatches Pillow's RGB.
4. **D. Human vision is most sensitive to green/brightness** — The 0.299/0.587/0.114 weights follow human luminance perception, where green dominates.
5. **C. Hue stays roughly constant under lighting changes while brightness moves to V** — Separating hue from brightness makes 'select this color' a clean band in H, robust to lighting.
6. **A. Box blur (average)** — Averaging each neighborhood smooths the image — a box blur.
7. **A. floor((n + 2p − k)/s) + 1** — This formula gives the spatial output size and is used constantly in CNN design.
8. **C. Cross-correlation (no kernel flip)** — Libraries skip the flip; for learned kernels it doesn't matter, so 'convolution' means correlation.
9. **C. Detect intensity gradients (edges)** — Sobel responds to intensity change in a direction — a gradient/edge detector.
10. **A. A 1-D horizontal pass then a 1-D vertical pass** — Separability turns a k×k kernel into two 1-D passes — 2k multiplies per pixel instead of k².
