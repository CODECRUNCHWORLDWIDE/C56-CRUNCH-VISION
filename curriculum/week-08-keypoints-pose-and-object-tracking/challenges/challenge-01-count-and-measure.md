# Challenge 1 — Count uniques and measure motion, with an error budget

Turn tracking into a real analytic — line-crossing counts and speed estimation, the way traffic
and sports systems do — and account rigorously for where the numbers go wrong.

1. On a clip with objects crossing a scene, use your tracker to **count unique objects** that cross a virtual
   line, each ID counted once. Show explicitly why this needs *tracking*, not per-frame detection (which
   multi-counts every object it sees each frame).
2. Estimate each object's **speed** in pixels/second from its track positions over time (finite-difference
   the Kalman-smoothed centre, or read the velocity state directly). If you can calibrate a real-world scale
   (a known distance in frame, or a homography to the ground plane), convert to real units and state your
   calibration assumptions.
3. **Error budget.** Decompose your count and speed error into contributions from missed detections (FN), ID
   swaps (IDSW), fragmentation, and drift. Tie each back to a metric from Lecture 3 (which of MOTA/IDF1/HOTA
   would flag it?) and estimate, roughly, how many miscounts each source caused on your clip.

**Deliverable:** an annotated clip with a running unique-count and per-object speeds, plus a written error
budget mapping each error source to a tracking failure mode and metric. Building a real metric on top of
tracking shows why identity-over-time is the hard, valuable part.
