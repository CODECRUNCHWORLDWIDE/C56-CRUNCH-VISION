# Challenge 2 — Map the video cost/accuracy frontier

Video forces brutal cost/accuracy trade-offs. Quantify them so you can make deployment
decisions instead of defaulting to the biggest model.

1. Take a video model and vary the knobs that trade cost for accuracy: **number of sampled frames**
   (e.g. 4 vs. 8 vs. 16 vs. 32), **input resolution**, and **clip sampling** (uniform vs. dense).
   Optionally include a rung change (frame-aggregation vs. (2+1)D vs. Transformer).
2. Measure **accuracy** (on a small labeled, video-split set) and **inference latency + peak memory**
   for each setting, on fixed hardware.
3. Plot accuracy vs. cost (latency or FLOPs). Identify the **Pareto frontier** and the point of
   diminishing returns. What is the cheapest setting that stays 'good enough' for your task?
4. Recommend a configuration for two scenarios: an **offline analytics job** (accuracy-first) and a
   **real-time edge camera** (latency/memory-first — a Week-11 preview). Justify each with the plot.

**Deliverable:** an accuracy-vs-cost plot with the Pareto frontier marked, across frame/resolution
settings, and a defended configuration recommendation for both scenarios. Making the cost trade-off
explicit — rather than defaulting to the biggest model — is exactly the engineering judgment video
demands.
