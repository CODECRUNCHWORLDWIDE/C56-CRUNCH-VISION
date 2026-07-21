# Week 9 — Quiz

Ten questions. Answer key below.

**1. Optical flow is:**

- A. A bounding box tracker
- B. The per-pixel apparent motion field between two frames
- C. A per-pixel class map
- D. An edge detector

**2. The brightness-constancy assumption states that:**

- A. Pixels get brighter over time
- B. Color is irrelevant
- C. The background is always black
- D. A moving physical point keeps roughly the same brightness across frames

**3. The aperture problem is that the flow equation:**

- A. Gives one equation for two unknowns per pixel, so motion along an edge is ambiguous
- B. Only works on video Transformers
- C. Requires color
- D. Has no solution ever

**4. Lucas–Kanade resolves the underdetermined flow by assuming:**

- A. Flow is constant in a small neighborhood
- B. No motion at all
- C. Global brightness change
- D. Every pixel is independent

**5. A single frame is insufficient for many actions because:**

- A. Color is missing
- B. CNNs can't see
- C. The action is defined by change/motion over time (e.g. sitting down vs. standing up)
- D. Frames are too large

**6. A 3D CNN differs from a 2D CNN by:**

- A. Convolving over time as well as space (a (t, h, w) kernel)
- B. Only using flow
- C. Using color
- D. Having no kernels

**7. A two-stream network feeds its temporal stream with:**

- A. Random noise
- B. Class labels
- C. Segmentation masks
- D. Optical flow (explicit motion)

**8. Video Transformers (e.g. TimeSformer) model time by:**

- A. Ignoring it
- B. Using only 2D convolutions
- C. Applying attention across space and time
- D. Averaging frames

**9. The most important honesty check for a video model is to:**

- A. Use the biggest model
- B. Train on all frames
- C. Skip evaluation
- D. Compare against a single-frame baseline to see if temporal modeling was worth its cost

**10. To avoid leakage, video datasets should be split:**

- A. Randomly per pixel
- B. By video (whole videos in one split)
- C. By frame
- D. By class only

---

## Answer key

1. **B. The per-pixel apparent motion field between two frames** — Flow gives each pixel a motion vector (u, v) between consecutive frames.
2. **D. A moving physical point keeps roughly the same brightness across frames** — This assumption yields the optical flow equation relating spatial and temporal gradients.
3. **A. Gives one equation for two unknowns per pixel, so motion along an edge is ambiguous** — Through a small window you sense only motion perpendicular to an edge; extra assumptions resolve it.
4. **A. Flow is constant in a small neighborhood** — A local-constancy window provides enough equations to solve for (u, v) at good points.
5. **C. The action is defined by change/motion over time (e.g. sitting down vs. standing up)** — Order and motion across frames carry the signal a single frame lacks.
6. **A. Convolving over time as well as space (a (t, h, w) kernel)** — 3D kernels learn spatiotemporal features directly — powerful but expensive.
7. **D. Optical flow (explicit motion)** — The flow stream provides motion so the network needn't learn it from raw pixels.
8. **C. Applying attention across space and time** — Spacetime attention captures long-range temporal dependencies — SOTA but costly.
9. **D. Compare against a single-frame baseline to see if temporal modeling was worth its cost** — If a frame baseline nearly matches it, the expensive temporal model wasn't needed.
10. **B. By video (whole videos in one split)** — Frames within a video are highly correlated; split by video, not frame.
