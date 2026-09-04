# Challenge 1 — Build something real with optical flow

Optical flow is a building block — use it to make a real effect or measurement, then break it
on purpose and explain the breakage with the math.

1. Pick one: **motion-triggered detection** (flag frames where significant motion occurs — a simple
   smart-camera trigger with a magnitude threshold and hysteresis), **video stabilization** (estimate
   global camera motion from flow, fit a similarity/affine transform, and warp frames to cancel it), or
   **frame interpolation** (synthesize an in-between frame by warping along flow for slow-motion).
2. Implement it with classical flow (Farneback / Lucas-Kanade), no training required. Optionally add a
   learned-flow variant (RAFT) and compare quality.
3. Test on real clips and **catalog where it breaks** — fast motion, occlusion, low texture, lighting
   changes, motion blur (each violates a specific assumption).
4. Explain **each failure** in terms of the flow assumptions from Lecture 1 (brightness constancy, the
   aperture problem, small-motion Taylor validity) and Lecture 4 (occlusion has no true match).

**Deliverable:** a working flow-based application/effect with a demo (before/after or triggered clips)
and a **failure analysis grounded in the brightness-constancy and aperture limitations**. Connecting the
math to the failures is the graded skill — a working demo with no failure analysis does not pass.
