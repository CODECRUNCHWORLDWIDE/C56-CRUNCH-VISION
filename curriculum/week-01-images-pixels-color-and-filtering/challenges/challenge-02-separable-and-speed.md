# Challenge 2 — Make convolution fast: separability and the FFT

Your hand-written triple loop is correct but painfully slow. Explore the three levers real
libraries pull — vectorization, separability, and the FFT — and the math behind each.

1. Time `convolve2d` on a large image vs. `cv2.filter2D`. Report the gap and attribute it (Python loop vs.
   vectorized C).
2. **Separability:** implement Gaussian blur as two 1-D passes and show it matches the full 2-D kernel. Count
   the multiplies per pixel (`2k` vs `k²`) and plot measured runtime vs. kernel size `k` for both — confirm
   the crossover grows with `k`.
3. **FFT convolution:** implement `f * h` via `IFFT(FFT(f)·FFT(h))` (with appropriate zero-padding to avoid
   circular-convolution wraparound). Verify it matches spatial convolution, then time it vs. the spatial
   method as kernel size grows. Find the kernel size where FFT overtakes direct convolution and explain the
   `O(N k²)` vs `O(N log N)` crossover.
4. Reason about why libraries push convolution to vectorized C and then the GPU, and connect it to why CNN
   training wants a GPU (many convolutions, massively parallel, arithmetic-bound).

**Deliverable:** a timing study (table + plots) covering vectorization, separability, and FFT convolution,
with the multiply/complexity counts explaining each crossover. Understanding *why* convolution is expensive
explains a large fraction of later engineering choices.
