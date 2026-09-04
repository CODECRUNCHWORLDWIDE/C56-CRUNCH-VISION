# Week 2 — Quiz

Fifteen questions spanning Canny's optimality criteria, the structure tensor, scale-space and SIFT internals, descriptors and matching, and robust two-view geometry. Attempt each before the answer key.

**1. Canny derived the optimal 1-D edge detector by simultaneously maximizing three criteria. They are:**

- A. brightness, contrast, and saturation
- B. good detection (SNR), good localization, and single response per edge
- C. scale, rotation, and affine invariance
- D. speed, memory, and accuracy

<details>
<summary>Answer</summary>

**B. good detection (SNR), good localization, and single response per edge** — Canny (1986) optimized detection SNR, localization accuracy, and a single response; the optimal filter is ~the first derivative of a Gaussian.

</details>

**2. In the Canny pipeline, non-maximum suppression realizes which of Canny's criteria?**

- A. the single-response criterion (one maximum per edge)
- B. scale invariance
- C. illumination invariance
- D. good detection

<details>
<summary>Answer</summary>

**A. the single-response criterion (one maximum per edge)** — NMS keeps only the local maximum along the gradient, thinning a ridge to one pixel — the discrete single-response criterion.

</details>

**3. Hysteresis thresholding keeps a weak edge pixel only if it:**

- A. lies at the image center
- B. is the brightest pixel in its row
- C. connects via an edge path to a strong edge pixel
- D. has zero gradient

<details>
<summary>Answer</summary>

**C. connects via an edge path to a strong edge pixel** — A weak edge continuing a strong one is likely real; isolated weak edges (noise) are dropped by the connectivity requirement.

</details>

**4. The second-moment (structure) matrix M used by Harris is defined as:**

- A. the image covariance of pixel intensities
- B. the Hessian of intensity ∇^2 I
- C. the Fourier transform of the patch
- D. a window-weighted sum of gradient outer products Σ w ∇I ∇I^T

<details>
<summary>Answer</summary>

**D. a window-weighted sum of gradient outer products Σ w ∇I ∇I^T** — M = Σ w [[Ix^2, IxIy],[IxIy, Iy^2]] is the windowed sum of gradient outer products; its eigenvalues give change energy in two directions.

</details>

**5. A pixel is classified as a corner when the eigenvalues of M satisfy:**

- A. both eigenvalues are large
- B. one eigenvalue is large and the other near zero
- C. both eigenvalues are near zero
- D. the eigenvalues are equal to the trace

<details>
<summary>Answer</summary>

**A. both eigenvalues are large** — Two large eigenvalues mean intensity changes in both directions — a corner. One large / one small is an edge; both small is flat.

</details>

**6. The Harris response R = det(M) - k·trace(M)^2 avoids computing eigenvalues because:**

- A. det(M) = λ1·λ2 and trace(M) = λ1+λ2 already encode the eigenvalue information
- B. eigenvalues of a 2x2 matrix are undefined
- C. it uses a neural network instead
- D. M is always diagonal

<details>
<summary>Answer</summary>

**A. det(M) = λ1·λ2 and trace(M) = λ1+λ2 already encode the eigenvalue information** — The determinant and trace are the two symmetric functions of the eigenvalues, so R detects 'both large' without a per-pixel eigen-solve.

</details>

**7. Harris corners are invariant to rotation but NOT to scale because:**

- A. R uses color, which changes with scale
- B. R depends only on eigenvalues (rotation-invariant) but the fixed window picks one scale
- C. rotation changes the eigenvalues
- D. Harris requires a GPU at large scale

<details>
<summary>Answer</summary>

**B. R depends only on eigenvalues (rotation-invariant) but the fixed window picks one scale** — Eigenvalues are rotation-invariant, but the fixed analysis window commits to a single scale, so corners are lost under zoom.

</details>

**8. Scale-space theory (Koenderink/Lindeberg) singles out the Gaussian kernel because it is:**

- A. the unique linear kernel that creates no new structure as scale increases (causality / heat equation)
- B. the fastest kernel to compute
- C. the kernel with the smallest support
- D. the only separable kernel

<details>
<summary>Answer</summary>

**A. the unique linear kernel that creates no new structure as scale increases (causality / heat equation)** — Causality axioms (no new extrema across scale) force the heat equation, whose Green's function is the Gaussian — it is the unique scale-space kernel.

</details>

**9. SIFT approximates the scale-normalized Laplacian-of-Gaussian using:**

- A. the Difference-of-Gaussian (subtracting adjacent scales of the Gaussian pyramid)
- B. a median filter
- C. the Harris response
- D. the gradient magnitude

<details>
<summary>Answer</summary>

**A. the Difference-of-Gaussian (subtracting adjacent scales of the Gaussian pyramid)** — From ∂G/∂t = (1/2)∇^2 G, G_{kσ}-G_σ ≈ (k-1)σ^2 ∇^2 G_σ, so DoG is a near-free approximation of the scale-normalized LoG.

</details>

**10. SIFT rejects edge-like DoG extrema using the principal-curvature ratio Tr(H)^2/det(H) — this is essentially:**

- A. the same 'change in two directions' idea as Harris, applied to the Hessian of D
- B. the ratio test
- C. hysteresis
- D. Hartley normalization

<details>
<summary>Answer</summary>

**A. the same 'change in two directions' idea as Harris, applied to the Hessian of D** — Lowe reuses the Harris/eigenvalue idea on the Hessian of D to discard points well-localized in only one direction (edges).

</details>

**11. The SIFT descriptor is a 128-D vector formed from:**

- A. a 4x4 grid of 8-bin gradient-orientation histograms (4·4·8 = 128)
- B. the 128 largest Fourier coefficients
- C. 128 binary intensity comparisons
- D. raw pixel intensities of a 128-pixel patch

<details>
<summary>Answer</summary>

**A. a 4x4 grid of 8-bin gradient-orientation histograms (4·4·8 = 128)** — SIFT concatenates 4x4 spatial cells each holding an 8-bin gradient-orientation histogram, then normalizes for illumination invariance.

</details>

**12. ORB descriptors are compared with the Hamming distance rather than L2 because they are:**

- A. color histograms
- B. 256-bit binary strings from intensity comparisons
- C. real-valued 128-D vectors
- D. probability distributions

<details>
<summary>Answer</summary>

**B. 256-bit binary strings from intensity comparisons** — ORB/BRIEF are binary strings; Hamming distance (XOR + popcount) is the correct, fast metric — using L2 would be wrong and slow.

</details>

**13. Lowe's ratio test keeps a match only if:**

- A. the two descriptors are bitwise identical
- B. it lies near the image center
- C. the colors match
- D. the nearest neighbour distance is clearly smaller than the second-nearest (d1 < ρ·d2)

<details>
<summary>Answer</summary>

**D. the nearest neighbour distance is clearly smaller than the second-nearest (d1 < ρ·d2)** — A distinctive match beats its runner-up; ambiguous points (two near-equal candidates) are rejected — removing most false matches.

</details>

**14. A planar homography relating two views has how many degrees of freedom, and how many point correspondences are minimally needed?**

- A. 8 DoF, 4 correspondences
- B. 9 DoF, 9 correspondences
- C. 6 DoF, 3 correspondences
- D. 5 DoF, 5 correspondences

<details>
<summary>Answer</summary>

**A. 8 DoF, 4 correspondences** — A 3x3 homography is defined up to scale (8 DoF); each correspondence gives 2 equations, so 4 points (no 3 collinear) suffice.

</details>

**15. RANSAC needs N ≥ log(1−p)/log(1−w^s) iterations. For s=4, inlier ratio w=0.5, and p=0.99, N is approximately:**

- A. about 7
- B. about 720
- C. about 4
- D. about 72

<details>
<summary>Answer</summary>

**D. about 72** — w^s = 0.5^4 = 0.0625; N ≥ log(0.01)/log(0.9375) ≈ 4.605/0.0645 ≈ 72 iterations.

</details>

---
