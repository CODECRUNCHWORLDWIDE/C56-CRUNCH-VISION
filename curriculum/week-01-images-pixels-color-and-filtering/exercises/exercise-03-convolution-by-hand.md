# Exercise 3 — Implement convolution and filter an image

**Goal:** implement the course's central operation yourself.

## Tasks

1. Write `convolve2d(img, kernel)` with explicit padding (Lecture 3). Do not use `cv2.filter2D` or `scipy` yet.
2. Apply three kernels to a grayscale image: a 3×3 box blur, a sharpen kernel, and a Sobel horizontal-gradient kernel. Display each result next to the original.
3. Verify your implementation against `cv2.filter2D` on the same kernel — the outputs should match (up to border handling). Explain any border differences.
4. Apply the Sobel kernel in both x and y, then compute gradient magnitude `sqrt(gx² + gy²)`. Display it — you have just built a crude edge detector, previewing Week 2.

## Deliverable

A notebook with your `convolve2d`, the three filtered outputs, the library cross-check, and the gradient-magnitude image. If your blur looks like noise or your edges are blank, your indexing or padding is off — debug it against the library.
