# Challenge 2 — Interrogate the hype

'Transformers replaced CNNs' is a headline, not an engineering fact. Test it critically.

1. Compare, on the same task and data budget, a modern ViT, a classic CNN (ResNet), and a modernized CNN (ConvNeXt) if available — matched for parameters and training recipe.
2. Hold the *training recipe* (augmentation, schedule, epochs) constant, then vary it, to show how much of any 'architecture' difference is really the training recipe (the ConvNeXt lesson).
3. Measure not just accuracy but inference latency and memory at your target resolution — including the ViT's N² cost as resolution rises.
4. Write a critical, evidence-based verdict: for *what* conditions is each architecture genuinely better?

**Deliverable:** a matched comparison and a critical write-up separating real architectural differences from training-recipe and hype. Being able to cut through fashion with measurement is a defining professional skill.
