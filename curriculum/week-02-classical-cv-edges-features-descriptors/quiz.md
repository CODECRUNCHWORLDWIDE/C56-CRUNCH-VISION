# Week 2 — Quiz

Ten questions. Answer key below.

**1. An edge in an image is fundamentally:**

- A. A location where intensity changes sharply (a large gradient)
- B. A region of constant color
- C. A corner
- D. The image border

**2. The first stage of the Canny pipeline is:**

- A. Gaussian blur to suppress noise
- B. Corner detection
- C. Descriptor matching
- D. Hysteresis

**3. Non-maximum suppression in Canny is used to:**

- A. Match features
- B. Increase brightness
- C. Thin thick edges to one-pixel-wide lines
- D. Detect corners

**4. Hysteresis thresholding keeps a weak edge pixel only if it:**

- A. Is very bright
- B. Is at the image center
- C. Has zero gradient
- D. Connects to a strong edge

**5. A corner differs from an edge because a corner:**

- A. Is brighter
- B. Has intensity change in two directions, making it locatable
- C. Is always at a keypoint
- D. Has no gradient

**6. The Harris detector distinguishes corners using:**

- A. Color histograms
- B. A neural network
- C. Fourier coefficients
- D. The two eigenvalues of the local structure tensor

**7. A keypoint adds to a corner:**

- A. A class label
- B. Scale and orientation for repeatability across zoom and rotation
- C. A bounding box
- D. A color

**8. A descriptor's purpose is to:**

- A. Summarize local appearance so the same point can be matched across images
- B. Locate the keypoint
- C. Threshold gradients
- D. Blur the region

**9. Lowe's ratio test keeps a match only if:**

- A. The descriptors are identical
- B. It is near the image center
- C. The colors match
- D. The best neighbor is far from the second-best (best < 0.75 × second)

**10. A key advantage of classical features over deep features is:**

- A. No training data needed, real-time on CPU, and interpretable
- B. Always higher accuracy
- C. They only work on faces
- D. They need a GPU

---

## Answer key

1. **A. A location where intensity changes sharply (a large gradient)** — Edges are high-gradient pixels — sharp intensity change.
2. **A. Gaussian blur to suppress noise** — Blurring first prevents noise speckles from becoming false edges.
3. **C. Thin thick edges to one-pixel-wide lines** — It keeps only local maxima along the gradient direction, thinning ridges.
4. **D. Connects to a strong edge** — Weak edges continuing a strong edge are likely real; isolated weak edges are dropped.
5. **B. Has intensity change in two directions, making it locatable** — Two-directional change makes a corner precisely locatable, unlike a slidable edge.
6. **D. The two eigenvalues of the local structure tensor** — Both eigenvalues large means change in both directions — a corner.
7. **B. Scale and orientation for repeatability across zoom and rotation** — Scale and orientation make the same physical point re-findable across transformations.
8. **A. Summarize local appearance so the same point can be matched across images** — It encodes what the point looks like as a comparable vector.
9. **D. The best neighbor is far from the second-best (best < 0.75 × second)** — An unambiguous match beats its runner-up clearly; ambiguous ones are rejected.
10. **A. No training data needed, real-time on CPU, and interpretable** — Classical methods are training-free, fast, and explainable — often the honest right tool.
