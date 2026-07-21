# Challenge 2 — Diagnose deliberately broken training

Debugging training is a skill you build by breaking things on purpose.

1. Create four broken training runs, each with exactly one injected fault: (a) learning rate 100× too high, (b) forgotten input normalization, (c) a train/test leak (same images in both), (d) a shuffled/wrong label mapping.
2. For each, capture the symptom in the loss/accuracy curves (NaN, no learning, suspiciously perfect validation, etc.).
3. Write a diagnostic guide mapping each *symptom* to its *cause* and *fix* — the table you'd want on the wall when a real run misbehaves.
4. Demonstrate the "overfit 10 images" sanity check catching one of the bugs.

**Deliverable:** the four broken runs with their curves and a symptom→cause→fix diagnostic guide. Being able to read failure is what separates practitioners who ship from those who guess.
