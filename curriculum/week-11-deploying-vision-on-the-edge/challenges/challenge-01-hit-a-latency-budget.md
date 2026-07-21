# Challenge 1 — Hit a hard latency budget

Real deployment means a hard constraint: 'must run at 30 fps (≤33 ms/image) on this device.' Meet it.

1. Set a concrete target (a latency and/or memory budget on your CPU or an edge device you have). Start from an accurate model that *misses* the budget.
2. Apply the toolkit to hit the budget: swap to an efficient architecture, reduce input resolution, quantize, and/or prune — measuring after each change.
3. Find the *most accurate* configuration that stays within the budget. Plot the accuracy-vs-latency path you took.
4. Report the final shipped model's held-out accuracy, latency (median and p95), and size.

**Deliverable:** a model meeting the stated budget, the optimization path with measurements, and the final honest numbers. Optimizing accuracy *subject to* a hard constraint — the real deployment problem — is the skill being tested.
