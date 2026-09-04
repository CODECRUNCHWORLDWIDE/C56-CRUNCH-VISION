# Challenge 2 — Diagnose deliberately broken training

Debugging training is a skill you build by breaking things on purpose. You will inject faults,
capture the symptom, and build the diagnostic table you would want on the wall when a real run misbehaves.

1. Create **five** broken training runs, each with exactly one injected fault: (a) learning rate ~100× too high;
   (b) forgotten input normalization; (c) a train/test leak (same or grouped images in both); (d) a
   shuffled/wrong label mapping; (e) `model.eval()` never called at evaluation so BatchNorm/dropout stay in
   training mode.
2. For each, capture the symptom in the loss/accuracy curves or the metrics (NaN, no learning, suspiciously
   perfect validation, a train/eval mismatch, etc.).
3. Write a diagnostic guide mapping each *symptom* → *cause* → *fix*, and for the LR case connect the NaN to the
   local `2/λ_max` step-size ceiling from Lecture 3/4.
4. Demonstrate the "overfit 10 images" sanity check catching one of the bugs (e.g. the label mapping), and
   explain why a healthy model should reach ~100% on a handful.

**Deliverable:** the five broken runs with their curves/metrics and a symptom→cause→fix diagnostic guide. Being
able to *read* failure is what separates practitioners who ship from those who guess.
