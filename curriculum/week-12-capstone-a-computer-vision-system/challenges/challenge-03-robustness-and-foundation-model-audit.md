# Challenge 3 — Robustness and foundation-model audit

An open, research-flavored challenge: stress-test your system against reality and against the
strongest available baseline.

1. **Corruption robustness.** Build a small corruption suite (blur, Gaussian noise, JPEG compression,
   brightness/contrast shifts, one weather effect) at a few severities, apply it to your test set, and plot
   your metric vs. severity for each corruption. Report the *relative* degradation and identify which
   corruption breaks your model first and why.
2. **Foundation-model gap.** Run a zero-shot foundation-model baseline (CLIP for classification, SAM for
   segmentation) on your task and compare it to your trained model *under the same corruptions*. Does your
   fine-tuned model degrade faster or slower than the zero-shot model? Foundation models are often more
   robust to shift — is that true here?
3. **Analyze.** Where does your model's clean-test advantage survive shift, and where does it evaporate?
   State a threat model: which of these shifts (or adversarial perturbations) actually matter for your
   deployment, and which don't?

**Deliverable:** a short report with the robustness curves, the foundation-model comparison under shift, and
a clear threat-model statement. Negative or partial results, well-analyzed, earn full credit — the graded
skill is experimental rigor and interpretation, not a headline win.
