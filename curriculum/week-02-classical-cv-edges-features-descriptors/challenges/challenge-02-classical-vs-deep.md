# Challenge 2 — When would you *not* use deep learning? Benchmark it.

Deep learning is not always the answer. Build the argument for classical vision with evidence, and
show honestly where learned features win.

1. Pick a concrete task where classical features plausibly win: real-time panorama stitching on a phone,
   matching two images with *no* labeled training data, or localizing a robot on a CPU under a tight
   latency budget.
2. **Time it.** Benchmark your ORB (and SIFT) match+RANSAC pipeline on an image pair — detection,
   description, matching, and fitting separately. Report frames-per-second on a CPU. Contrast with the
   compute, data, and latency a learned matcher (SuperPoint+SuperGlue or LoFTR) would require to train and
   run.
3. **Find where classical loses.** Construct or find a *hard* pair — wide baseline, low texture (a blank
   wall, sky), or day-vs-night — where ORB/SIFT produce few or no correct matches. Explain, from Lecture 5,
   why detector-based classical methods struggle here and why a detector-free learned matcher (LoFTR) helps.
4. Write a decision guide: given constraints (data availability, latency, hardware, explainability, scene
   difficulty), when do you reach for classical features vs. a neural matcher?

**Deliverable:** a one-page decision guide backed by your timing numbers *and* a documented failure case
where classical loses. The mark of an engineer is choosing the right tool honestly with evidence, not
defaulting to either the fashionable or the nostalgic one.
