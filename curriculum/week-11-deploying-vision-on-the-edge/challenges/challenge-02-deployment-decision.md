# Challenge 2 — Make and defend a deployment decision with a decision matrix

Deployment is a judgment call across competing objectives. Make one rigorously and defend it.

1. Pick a realistic scenario with specific constraints and *rank* them: e.g. a battery-powered wildlife camera
   (power + peak-memory tight, latency loose), a real-time AR phone app (tail latency tight), a surgical/medical
   screening device (accuracy + failure-mode-safety first), or a cloud-fallback hybrid.
2. Evaluate >=4 candidate configurations (architecture x precision x resolution, incl. at least one QAT and one
   PTQ point) on accuracy, median+p99 latency, peak memory, size, and an *estimated* energy/inference (use
   DRAM-traffic reasoning from Lecture 1 if you cannot measure joules).
3. Build a weighted **decision matrix** whose weights reflect *that scenario's* ranked constraints, and
   recommend one configuration, showing the arithmetic — not just an assertion.
4. State what you would measure on the real target before shipping, and the risks: preprocessing parity,
   distribution shift, tail-latency safety, and the production monitoring you would set up (a nod to
   [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/)). If the scenario is safety- or privacy-sensitive, say explicitly
   how on-device inference and quantized-model re-validation address it.

**Deliverable:** a decision matrix with justified weights and a defended recommendation for the scenario, plus
a pre-ship checklist and risk register. Justifying a choice against ranked constraints — not just reporting
numbers — is what turns benchmarks into engineering.
