# Challenge 2 — Interrogate the hype: architecture vs. recipe vs. cost

"Transformers replaced CNNs" is a headline, not an engineering fact. Test it critically and
quantitatively.

1. On the same task and data budget, compare a modern ViT, a classic CNN (ResNet-50), and a modernized CNN
   (ConvNeXt-T if available) — **matched for parameter count and FLOPs**.
2. **Isolate the recipe.** Hold the training recipe (augmentation, schedule, epochs, optimizer) constant
   across all three, then deliberately vary it, to quantify how much of any "architecture" difference is
   really the recipe (the ConvNeXt lesson, Lecture 3). Report accuracy under a weak recipe and a strong one.
3. **Measure the full cost.** Report not just accuracy but inference latency and peak memory at your target
   resolution, and *sweep resolution* (224, 384, 512) to expose the ViT's `N^2` blow-up versus the CNN's
   linear growth. Include a throughput (images/sec) number.
4. Write a critical, evidence-based verdict: under *what* conditions (data scale, resolution, latency
   budget, pretraining availability) is each architecture genuinely better? No fashion, only measurement.

**Deliverable:** a matched, recipe-controlled comparison table plus resolution-swept latency/memory curves,
and a critical write-up separating real architectural differences from training-recipe effects and hype.
Cutting through fashion with measurement is a defining professional skill.
