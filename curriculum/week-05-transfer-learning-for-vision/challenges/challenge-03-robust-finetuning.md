# Challenge 3 — Fine-tune without wrecking robustness (LP-FT and WiSE-FT)

This challenge targets the modern failure mode from Lecture 5: fine-tuning a strong pretrained
backbone can *improve* in-distribution accuracy while *destroying* out-of-distribution robustness. Reproduce
the effect and defeat it.

1. Start from a robust backbone — CLIP, or a strong self-supervised model. Establish an in-distribution (ID)
   test set and an out-of-distribution (OOD) test set for the same classes (e.g. a different image style,
   corruption, or a shifted source). For CLIP, record the **zero-shot** ID and OOD accuracy as a baseline.
2. **Naive fine-tune.** Full fine-tune from a random head. Measure ID and OOD accuracy. Show (if it appears) the
   characteristic pattern: ID up, OOD down relative to zero-shot — feature distortion (Kumar et al., 2022).
3. **LP-FT.** Linear-probe to a good head first, then fine-tune. Re-measure ID and OOD. Report the OOD recovery.
4. **WiSE-FT.** Interpolate weights `theta = (1-alpha) theta_zeroshot + alpha theta_finetuned` for a sweep of
   `alpha in [0,1]`. Plot ID and OOD accuracy vs. `alpha` and find the interpolation that best trades them
   (Wortsman et al., 2022).
5. Analyze: which method gave the best ID/OOD frontier, and what does that say about treating pretrained weights
   as a resource to preserve rather than overwrite?

**Deliverable:** an ID-vs-OOD results table (zero-shot, naive FT, LP-FT) plus the WiSE-FT `alpha`-sweep plot, and
a written account of the robustness/accuracy trade-off you observed. Negative or partial results, well analyzed,
earn full credit — the graded skill is rigorous measurement of the trade-off, not confirming a headline.
