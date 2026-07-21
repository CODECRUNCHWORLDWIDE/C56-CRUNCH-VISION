# Challenge 1 — Build something with optical flow

Optical flow is a building block — use it to make a real effect or measurement.

1. Pick one: **motion-triggered detection** (flag frames where significant motion occurs — a simple smart-camera trigger), **video stabilization** (estimate global camera motion from flow and warp frames to cancel it), or **frame interpolation** (synthesize an in-between frame using flow for slow-motion).
2. Implement it with classical flow (Farnebäck/Lucas–Kanade), no training required.
3. Test on real clips and catalog where it breaks — fast motion, occlusion, low texture, lighting changes (all violate brightness constancy).
4. Explain each failure in terms of the flow assumptions from Lecture 1.

**Deliverable:** a working flow-based application/effect with a demo and a failure analysis grounded in the brightness-constancy and aperture limitations. Connecting the math to the failures is the skill.
