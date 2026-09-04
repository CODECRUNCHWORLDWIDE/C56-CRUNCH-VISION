# Exercise 3 — Implement convolution, verify separability, and cross-check

**Goal:** implement the course's central operation yourself and verify its key properties.

## Tasks

1. Write `convolve2d(img, kernel, stride=1)` with explicit padding (Lecture 3). Do not use `cv2.filter2D` or
   `scipy` yet. Handle both `mode="edge"` and `mode="zero"` padding and report the visual difference at
   borders.
2. Apply a 3×3 box blur, a sharpen kernel, and a Sobel horizontal-gradient kernel to a grayscale image;
   display each next to the original.
3. **Separability:** build a 1-D Gaussian, form the full 2-D Gaussian by outer product, and blur two ways —
   the full 2-D convolution vs. two 1-D passes (rows then columns). Show the outputs match to numerical
   precision and count the multiplies each way (`k²` vs `2k`) for your kernel size.
4. Verify `convolve2d` against `cv2.filter2D` on one kernel (remember cv2 does correlation; flip if needed).
   Explain any border-handling differences.
5. Apply Sobel in x and y, compute gradient magnitude `sqrt(gx² + gy²)` and orientation `atan2(gy, gx)`;
   display both. You have built a crude edge detector previewing Week 2.

## Deliverable

A notebook with your `convolve2d`, the three filtered outputs, the separable-vs-full Gaussian equality plus
multiply counts, the library cross-check, and the gradient magnitude+orientation images. If your blur looks
like noise or edges are blank, your indexing/padding is off — debug against the library.
