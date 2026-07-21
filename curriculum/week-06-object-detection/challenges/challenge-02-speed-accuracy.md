# Challenge 2 — The speed–accuracy frontier

Detection is defined by the speed-vs-accuracy trade-off. Map it with real measurements.

1. Run at least two detectors on the same image set: one two-stage (Faster R-CNN) and one one-stage (a YOLO variant), plus different sizes of one family if available.
2. Measure each model's inference latency (ms/image on your hardware) and its mAP on your labeled set.
3. Plot mAP vs. latency — the frontier. Which model would you pick for: a real-time video app, an offline archival tagging job, a battery-powered edge camera (a Week-11 preview)?
4. Discuss how input resolution trades accuracy for speed within a single model.

**Deliverable:** a mAP-vs-latency plot and a written model-selection recommendation for three deployment scenarios. Choosing on the frontier — not just grabbing the highest-mAP model — is the real engineering judgment.
