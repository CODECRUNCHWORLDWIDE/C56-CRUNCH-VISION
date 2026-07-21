# Challenge 2 — Make and defend a deployment decision

Deployment is a judgment call across competing objectives. Make one rigorously and defend it.

1. Pick a realistic scenario with specific constraints: e.g. a battery-powered wildlife camera (power + memory tight, latency loose), a real-time AR phone app (latency tight), or a cloud API (accuracy-first, cost-aware).
2. Evaluate 3+ candidate configurations (architecture × quantization × resolution) on accuracy, latency, memory, size, and (if you can estimate) power.
3. Build a decision matrix and recommend one configuration, explicitly weighing the trade-offs for *that* scenario's constraints.
4. State what you'd measure on the real target before shipping, and the risks (preprocessing parity, real-world distribution shift, the accuracy you'd monitor in production — a nod to [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/)).

**Deliverable:** a decision matrix and a defended recommendation for the scenario. Being able to justify a deployment choice against constraints — not just report numbers — is what turns benchmarks into engineering.
