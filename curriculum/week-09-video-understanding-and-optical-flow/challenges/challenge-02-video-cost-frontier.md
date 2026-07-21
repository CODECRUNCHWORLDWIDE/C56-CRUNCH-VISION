# Challenge 2 — Map the video cost frontier

Video forces brutal cost/accuracy trade-offs. Quantify them so you can make deployment decisions.

1. Take a video model and vary the knobs that trade cost for accuracy: number of sampled frames (e.g. 4 vs. 8 vs. 16 vs. 32), input resolution, and clip length.
2. Measure accuracy (on a small labeled set) and inference latency/memory for each setting.
3. Plot accuracy vs. cost. Where are the diminishing returns? What's the cheapest setting that stays 'good enough'?
4. Recommend a configuration for two scenarios: an offline analytics job (accuracy-first) and a real-time edge camera (latency/memory-first, a Week-11 preview).

**Deliverable:** an accuracy-vs-cost plot across frame/resolution settings and a configuration recommendation for both scenarios. Making the cost trade-off explicit — rather than defaulting to the biggest model — is exactly the engineering judgment video demands.
