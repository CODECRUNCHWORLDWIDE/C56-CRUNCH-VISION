# Challenge 2 — Map the speed-accuracy frontier and pick on it

Detection is defined by the speed-vs-accuracy trade-off. Map it with real measurements and defend a
model choice — the actual engineering skill, which is *not* "grab the highest mAP."

1. **Run at least three detectors** on the same image set: one two-stage (Faster R-CNN), one one-stage (a YOLO
   variant), and one more point (a different size of one family, or RetinaNet). If you can, add a DETR-family
   model for contrast.
2. **Measure** each model's inference latency (warm the model; time only the forward pass; report mean and p95
   ms/image on stated hardware) and its mAP@[.5:.95] on your labeled set from Exercise 3.
3. **Plot the frontier:** mAP vs. latency, one point per model. Identify which models are **Pareto-dominated**
   (worse on both axes than some other model) and drop them from consideration.
4. **Recommend** a model for each of three scenarios, justifying from the plot: (a) a real-time video app
   (latency-bound), (b) an offline archival tagging job (accuracy-bound), (c) a battery-powered edge camera (a
   Week-11 preview: latency *and* energy/model-size bound).
5. **Resolution knob.** For one model, sweep input resolution (e.g. 512 / 800 / 1024 short side) and show how
   it trades mAP for latency *within a single model* — often a cheaper lever than switching models.

**Deliverable:** a mAP-vs-latency frontier plot with Pareto-dominated models marked, a resolution-sweep curve
for one model, and a written model-selection recommendation for the three scenarios. Choosing *on the frontier*
— not just grabbing the top-mAP model — is the graded judgment.
