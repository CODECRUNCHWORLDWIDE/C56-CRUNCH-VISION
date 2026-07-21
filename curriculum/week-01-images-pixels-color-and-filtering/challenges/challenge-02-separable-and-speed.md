# Challenge 2 — Make convolution fast

Your hand-written triple loop is correct but painfully slow. Explore why real libraries are fast — and the math that makes some kernels cheap.

1. Time your `convolve2d` on a large image vs. `cv2.filter2D`. Report the speed gap.
2. A Gaussian blur is **separable**: a 2-D Gaussian equals a 1-D Gaussian applied horizontally then vertically. Implement it as two 1-D passes and show it produces (nearly) the same result as the full 2-D kernel — and count the multiplications each way to explain the speedup for a `k×k` kernel (`2k` vs `k²` per pixel).
3. Reason about why libraries push convolution to optimized C / vectorized code and, later, to the GPU — and connect this to why CNN training wants a GPU.

**Deliverable:** a timing table and a short explanation of separability with the multiply counts. Understanding *why* convolution is expensive explains a lot of later engineering choices.
