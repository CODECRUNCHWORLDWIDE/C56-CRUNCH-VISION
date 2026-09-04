# Challenge 3 — Denoise without destroying edges

This is an open, research-flavored challenge on the central tension of Lecture 5: removing noise
while keeping edges. There is no single right answer — you make the design calls and defend them.

**Setup.** Take a clean image and add two kinds of noise separately: **Gaussian** noise (additive, normal)
and **salt-and-pepper** (impulse) noise, at a couple of severities. These noise types have different
statistics and reward different filters.

**Investigate.** Denoise each with at least: (a) a Gaussian blur, (b) a median filter, and (c) a bilateral
filter (implement it from scratch — the two-Gaussian weighting from Lecture 5 — or justify a library call).
For the bilateral, sweep the spatial `σ_s` and range `σ_r` and describe their distinct roles.

**Measure.** Report a quantitative metric against the clean original — PSNR and, better, SSIM (Wang et al.,
2004, "Image Quality Assessment: From Error Visibility to Structural Similarity," IEEE TIP) — for each
filter × noise combination. A single number hides edge damage; also show crops at edges.

**Analyze.** Which filter wins on Gaussian noise? On salt-and-pepper? Why does the median dominate impulse
noise (order statistics) while the bilateral dominates edge preservation under Gaussian noise? Where does each
fail (median eroding corners, bilateral leaving mottling, Gaussian smearing everything)? Relate your findings
to the linear-can't-separate-edges-from-noise argument.

**Deliverable:** a report with a filter × noise metric table (PSNR/SSIM), edge crops, your from-scratch
bilateral with the `σ_s`/`σ_r` sweep, and an honest account of when each filter wins and why. Well-analyzed
partial or negative results earn full credit — the graded skill is experimental rigor and interpretation.
