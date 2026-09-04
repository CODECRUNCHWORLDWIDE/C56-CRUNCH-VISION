# Challenge 2 — Demo and defend to a review panel

Prepare to present your capstone as you would to a hiring panel *and* an ethics review board.

1. Record or write a short **demo**: send a real image to your served `/predict` API and show the
   prediction (label + confidence, boxes, or mask overlaid), with the golden test demonstrating
   preprocessing parity.
2. Prepare answers to the questions a reviewer *will* ask: Why this task and architecture? How do you know
   the win isn't sampling noise (interval + paired test)? Are the probabilities calibrated (ECE)? Where does
   it fail, and for whom (subgroup audit, worst-group metric)? How brittle is it to shift, and what's your
   threat model? Where may it *not* be deployed, legally and ethically? What would you monitor after launch?
3. Write a two-part **defense**: one paragraph answering the toughest *technical* critique (statistics,
   leakage, calibration, robustness) and one answering the toughest *ethical/legal* critique (bias, consent,
   prohibited use), naming the specific concern and your specific answer.

**Deliverable:** the demo plus the two-part written defense. Being able to *defend* your choices —
statistically, ethically, and legally — is what turns a project into a job offer.
