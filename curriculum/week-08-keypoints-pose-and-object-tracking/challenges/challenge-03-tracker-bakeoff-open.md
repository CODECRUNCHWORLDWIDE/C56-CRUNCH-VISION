# Challenge 3 — Tracker bake-off: SORT vs. Deep SORT vs. ByteTrack (open)

An open, research-flavoured challenge. There is no universal best tracker; the right choice depends
on object count, occlusion, latency, and data. Your task is to *establish*, with evidence, which tracker wins
on a scenario you choose — and to explain why in terms of the mechanisms from Lectures 4 and 5.

**Setup.** Pick or record 2-3 clips spanning regimes: (i) few, well-separated objects; (ii) frequent
crossings/occlusion; (iii) a crowded scene. Obtain ground-truth identities (hand-label a subset, or use a
public MOT snippet). Implement or wire up three trackers on a *shared* detector: **SORT** (Kalman + IoU +
Hungarian), **Deep SORT** (add appearance + Mahalanobis gating), and **ByteTrack** (two-pass low-score
association, no appearance).

**Measure.** Report HOTA (with DetA/AssA), IDF1, MOTA, ID switches, and frames/second for each tracker on
each clip. Present a table and at least one annotated failure clip per tracker.

**Analyze.** Explain the results mechanistically: *why* does appearance help (or not) on crossings? *Why*
does ByteTrack's low-score pass recover occluded objects, and where does it add false tracks? Where does the
constant-velocity Kalman assumption break, and does it show up as drift (DetA) or swaps (AssA)? When would you
reach for a transformer tracker instead, and what would it cost you?

**Deliverable:** a short report with the metrics table, failure clips, and a defended recommendation for each
regime. Negative or surprising results, well-analyzed, earn full credit — the graded skill is experimental
rigor and mechanistic explanation, not crowning the newest method.
