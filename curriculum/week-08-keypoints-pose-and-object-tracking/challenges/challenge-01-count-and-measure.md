# Challenge 1 — Count objects and measure motion

Turn tracking into a real analytic — counting or speed estimation — the way traffic and sports systems do.

1. On a clip with objects crossing a scene, use your tracker to **count unique objects** that cross a virtual line (each ID counted once). Show why this needs *tracking*, not per-frame detection (which would multi-count).
2. Estimate each object's **speed** in pixels/second from its track positions over time. If you can calibrate a real-world scale (a known distance in frame), convert to real units.
3. Report where counting/speed errors come from — missed detections, ID swaps, drift — and tie each back to a tracking failure mode.

**Deliverable:** an annotated clip with a running count and per-object speeds, plus an error analysis. Building a real metric on top of tracking shows why identity-over-time is the hard, valuable part.
