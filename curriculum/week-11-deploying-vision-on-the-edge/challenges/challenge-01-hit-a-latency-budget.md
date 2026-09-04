# Challenge 1 — Hit a hard latency budget (constrained optimization)

Real deployment means a hard constraint: "must run at 30 fps (<=33 ms/image, p99) on this device." Meet
it — this is the actual deployment problem, maximize accuracy *subject to* a ceiling.

1. Set a concrete target: a latency (and optionally memory) budget on your CPU or a real edge device you have.
   Start from an accurate model that *misses* the budget.
2. Apply the toolkit and **measure the p99 after each change**: swap to an efficient architecture, reduce input
   resolution, structured-prune, and quantize (PTQ, then QAT if needed). Respect the roofline — do not spend
   effort cutting FLOPs on a memory-bound layer.
3. Find the *most accurate* configuration whose **p99** stays within budget (not just the median). Plot the
   accuracy-vs-latency path you took through configuration space, marking the feasible region.
4. Report the final shipped model's held-out accuracy, latency (median and p99), peak memory, and size — all
   on the exported, quantized artifact on the target.

**Deliverable:** a model meeting the stated budget at the *tail*, the optimization path with measurements at
each step, and the final honest numbers on the shipped artifact. Optimizing accuracy subject to a hard tail-
latency constraint — not just improving the average — is the skill being tested.
