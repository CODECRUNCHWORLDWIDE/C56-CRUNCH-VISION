# Challenge 3 — Temporal action localization with an ethics gate

Classification says *what* action; localization says *when*. Build a simple temporal
localizer and confront its surveillance implications head-on.

1. On a few longer clips, run your action model over a **sliding temporal window** to produce a score
   time-series per action class. Threshold and merge to propose **action segments** (start, end, label).
2. Evaluate with **temporal IoU** against hand-marked ground-truth segments on 2-3 clips: compute tIoU
   per proposal and report precision/recall at a tIoU threshold (e.g. 0.5). Discuss why the boundaries
   are genuinely ambiguous (when does 'standing up' start?).
3. Analyze failure: over-segmentation, boundary jitter, and confusion between motion-similar actions.
4. **Ethics gate (required to pass).** Write a one-page assessment: who could be surveilled by this,
   what consent/lawful basis a real deployment would need, how dataset bias (geographic/demographic
   skew of Kinetics) would produce unequal error rates, and a concrete decision on whether *you* would
   deploy it and under what constraints. Apply the mirror test (Lecture 3).

**Deliverable:** a working sliding-window temporal localizer with a tIoU evaluation on labeled
segments, a failure analysis, and a substantive ethics assessment. Both the tIoU numbers and the ethics
gate are graded; a strong localizer with a hand-wave ethics note does not pass.
