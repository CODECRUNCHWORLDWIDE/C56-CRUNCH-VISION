# Week 9 — Quiz

Fifteen questions spanning the flow constraint and aperture problem, the structure tensor and variational smoothness, RAFT's correlation volume, the architecture ladder, spacetime-attention cost, self-supervised video, and honest evaluation. Attempt each before the answer key.

**1. The optical flow constraint equation `I_x·u + I_y·v + I_t = 0` is derived by:**

- A. Taking the Fourier transform of two frames
- B. Solving the heat equation on the image
- C. First-order Taylor expansion of the brightness-constancy assumption
- D. Minimizing cross-entropy between frames

<details>
<summary>Answer</summary>

**C. First-order Taylor expansion of the brightness-constancy assumption** — Brightness constancy `I(x,y,t)=I(x+u,y+v,t+dt)` expanded to first order and simplified yields `∇I·(u,v)+I_t=0`.

</details>

**2. The aperture problem means the flow constraint at a single pixel:**

- A. Has no solution under any circumstances
- B. Requires a Transformer to solve
- C. Only applies to color images
- D. Gives one equation for two unknowns, fixing only motion normal to an edge

<details>
<summary>Answer</summary>

**D. Gives one equation for two unknowns, fixing only motion normal to an edge** — One scalar equation, two unknowns per pixel: only the component along `∇I` (normal to the edge) is determined; along-edge motion is invisible.

</details>

**3. Lucas-Kanade recovers flow by assuming flow is constant in a window and solving least squares; flow is well-determined exactly when the 2x2 structure tensor `M`:**

- A. Has one large and one near-zero eigenvalue
- B. Is diagonal
- C. Has both eigenvalues large (a corner)
- D. Is the zero matrix

<details>
<summary>Answer</summary>

**C. Has both eigenvalues large (a corner)** — `M` invertible with two large eigenvalues = a well-textured corner; one small eigenvalue is the aperture problem (an edge), both small is a flat region.

</details>

**4. Horn-Schunck differs from Lucas-Kanade by adding:**

- A. A larger convolution kernel only
- B. A softmax over classes
- C. A global smoothness prior `α²(‖∇u‖²+‖∇v‖²)` minimized as a variational energy
- D. Color channels

<details>
<summary>Answer</summary>

**C. A global smoothness prior `α²(‖∇u‖²+‖∇v‖²)` minimized as a variational energy** — Horn-Schunck minimizes a data term plus a global smoothness penalty; its Euler-Lagrange equations propagate motion into textureless/ambiguous pixels.

</details>

**5. Coarse-to-fine image pyramids are used in classical flow primarily to:**

- A. Compress the video
- B. Add temporal attention
- C. Handle large motion, since the small-displacement Taylor expansion fails for fast motion
- D. Reduce color banding

<details>
<summary>Answer</summary>

**C. Handle large motion, since the small-displacement Taylor expansion fails for fast motion** — The flow constraint's Taylor expansion is only valid for small displacements; on a downsampled image large motion becomes small, then is refined via warping.

</details>

**6. RAFT's central innovation over PWC-Net is:**

- A. An all-pairs 4-D correlation volume plus a recurrent GRU that refines flow iteratively
- B. Training without any data
- C. Using only Lucas-Kanade internally
- D. Removing all convolutions

<details>
<summary>Answer</summary>

**A. An all-pairs 4-D correlation volume plus a recurrent GRU that refines flow iteratively** — RAFT (Teed & Deng 2020) builds an all-pairs correlation volume once so no displacement is lost, then a shared-weight GRU refines residual flow — a learned iterative optimizer.

</details>

**7. Learned optical flow models are trained mostly on synthetic data (FlyingThings, Sintel) because:**

- A. Synthetic video is cheaper to store
- B. Networks cannot read real images
- C. Real video has no motion
- D. Per-pixel ground-truth flow is essentially impossible to annotate by hand on real video

<details>
<summary>Answer</summary>

**D. Per-pixel ground-truth flow is essentially impossible to annotate by hand on real video** — You cannot label a subpixel motion vector at every pixel by hand; renderers give perfect flow labels for free, and the learned features transfer to real footage.

</details>

**8. Why is a single frame insufficient for many action-recognition tasks?**

- A. Frames are too high-resolution
- B. CNNs cannot process RGB
- C. The action is defined by change/order over time (e.g. sitting down vs. standing up)
- D. Color is missing from single frames

<details>
<summary>Answer</summary>

**C. The action is defined by change/order over time (e.g. sitting down vs. standing up)** — Motion-defined actions look identical or ambiguous in one frame; the signal lives in the order and motion across frames.

</details>

**9. A 3D CNN differs from a 2D CNN by:**

- A. Convolving over time as well as space with a `(t,h,w)` kernel
- B. Requiring grayscale
- C. Removing all pooling
- D. Using only optical flow input

<details>
<summary>Answer</summary>

**A. Convolving over time as well as space with a `(t,h,w)` kernel** — 3D kernels slide over stacked frames to learn spatiotemporal features directly (C3D, I3D) — powerful but with an extra temporal axis multiplying FLOPs and memory.

</details>

**10. The (2+1)D / R(2+1)D factorization (Tran et al. 2018) replaces a 3D conv with:**

- A. A single fully-connected layer
- B. Two identical 3D convs
- C. A 2D spatial conv followed by a 1D temporal conv, with a nonlinearity between
- D. Optical flow preprocessing

<details>
<summary>Answer</summary>

**C. A 2D spatial conv followed by a 1D temporal conv, with a nonlinearity between** — Factorizing into spatial-then-temporal convolution retains most 3D expressive power at lower cost, adds a nonlinearity, and optimizes more easily.

</details>

**11. In a two-stream network (Simonyan & Zisserman 2014), the temporal stream is fed:**

- A. Class labels
- B. Segmentation masks
- C. Random noise
- D. Stacked optical flow (explicit motion)

<details>
<summary>Answer</summary>

**D. Stacked optical flow (explicit motion)** — The temporal stream consumes optical flow so the network gets explicit motion and need not learn it from raw RGB; the spatial stream handles appearance.

</details>

**12. Joint space-time self-attention over a video is expensive because its cost scales as:**

- A. Quadratic in the number of spacetime tokens `N·T`, i.e. `O((NT)²)`
- B. Logarithmic in clip length
- C. Constant regardless of resolution
- D. Linear in the number of frames only

<details>
<summary>Answer</summary>

**A. Quadratic in the number of spacetime tokens `N·T`, i.e. `O((NT)²)`** — Attention is quadratic in token count; with `N·T` spacetime tokens, doubling resolution or clip length quadruples compute — the blow-up factorization addresses.

</details>

**13. TimeSformer's 'divided space-time attention' reduces cost by:**

- A. Dropping half the frames at random
- B. Skipping attention entirely
- C. Attending spatially within a frame and temporally across frames separately, `O(TN²+NT²)`
- D. Using 1x1 convolutions instead of attention

<details>
<summary>Answer</summary>

**C. Attending spatially within a frame and temporally across frames separately, `O(TN²+NT²)`** — Divided attention factorizes the quadratic joint cost into separate spatial and temporal passes, far cheaper while still mixing information across spacetime over stacked blocks.

</details>

**14. VideoMAE masks ~90-95% of spacetime patches (a much higher ratio than image MAE) because:**

- A. Labels require it
- B. Higher masking uses more memory
- C. Video is highly redundant across time, so a low mask ratio lets the model cheat by copying neighboring frames
- D. GPUs cannot handle low mask ratios

<details>
<summary>Answer</summary>

**C. Video is highly redundant across time, so a low mask ratio lets the model cheat by copying neighboring frames** — Adjacent frames are near-duplicates; only very high masking makes reconstruction non-trivial and forces the model to learn genuine spatiotemporal structure.

</details>

**15. The single most important honesty check when evaluating a video model is to:**

- A. Always pick the largest available model
- B. Train on the test set for more accuracy
- C. Compare against a single-frame baseline, and split the data by whole video (not by frame)
- D. Report only top-1 accuracy

<details>
<summary>Answer</summary>

**C. Compare against a single-frame baseline, and split the data by whole video (not by frame)** — A frame baseline reveals whether temporal modeling earned its cost; splitting by video (not frame) prevents highly-correlated frames leaking the test set into training.

</details>

---
