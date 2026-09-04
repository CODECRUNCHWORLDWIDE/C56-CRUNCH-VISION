# Week 5 — Graduate Problem Set: Transferability, Fine-Tuning, and Domain Adaptation

Ten problems, easy to hard, mixing derivation, computation, and open analysis. Solution sketches are
at the end — attempt each fully before reading them. Notation: `phi` is a (possibly frozen) feature map, `H` a
hypothesis class of heads on `phi`, `D_S`/`D_T` the source/target distributions, `eps_S`/`eps_T` the source/
target errors.

**P1 (feature hierarchy).** Yosinski et al. (2014) transfer the first `k` layers of a source-trained network to
a target and retrain the rest. Sketch qualitatively how (a) transfer-with-freezing accuracy and (b) transfer-
with-fine-tuning accuracy vary with `k` from 1 to the full depth, and explain the crossover in terms of
*specificity* and *co-adaptation*.

**P2 (caching speedup).** A frozen backbone costs `C` FLOPs per image forward pass; a linear head costs `c << C`.
For a dataset of `N` images trained for `E` epochs, write the total forward FLOPs for (a) fine-tuning the whole
network and (b) linear-probing cached features (backbone run once). Give the speedup ratio and its limit as
`E -> infinity`.

**P3 (preprocessing sensitivity).** You feed a backbone trained with normalization `(mu, sigma)` an input
normalized instead with `(0, 1)`. For the very first convolution `w * x`, express the resulting activation in
terms of the intended one, and argue why the error compounds through depth. Why is this failure *silent*
(training still "works")?

**P4 (which quadrant).** For each scenario, name the quadrant and the recommended strategy, and justify: (a)
400 photos of 5 dog breeds; (b) 200k labeled chest X-rays; (c) 300 aerial-imagery tiles, 8 land-use classes;
(d) 1M labeled sketches. Where does parameter-efficient fine-tuning change your answer?

**P5 (H-divergence estimate).** You train a logistic-regression domain classifier to separate source features
from target features and obtain balanced accuracy `a`. Give the corresponding estimate of `d_{H Delta H}` (recall
its relation to `2(1 - 2*error)` forms), and state what `a = 0.5` and `a = 1.0` each imply about transfer
feasibility with the frozen `phi`.

**P6 (reading the bound).** Using `eps_T(h) <= eps_S(h) + (1/2) d_{H Delta H} + lambda`, you observe: `eps_S`
small, `d_{H Delta H}` large, `eps_T` large. Which lever(s) should you pull, and which experiment tells you
whether `lambda` is the true obstacle? Contrast the action you take if `lambda` turns out large vs. small.

**P7 (covariate vs. concept shift).** Give a concrete vision example of each: (a) covariate shift where feature
alignment fixes transfer, and (b) concept shift where it cannot. For (b), explain formally why `lambda` is large
and no frozen-`phi` head can be good on both domains.

**P8 (catastrophic forgetting, quantitative).** Model one weight `w` starting at a good value `w*` with gradient
noise `g_t` of scale `s`. Under SGD `w_{t+1} = w_t - eta g_t`, give the expected squared displacement from `w*`
after `T` steps as a function of `eta`, `s`, `T`. Use it to argue why warmup and a small LR prevent early
destruction of pretrained weights, and why layer-wise decay protects early layers most.

**P9 (linear probe vs. fine-tune, OOD).** Explain, following Kumar et al. (2022), why fine-tuning from a random
head can lower OOD accuracy even while raising ID accuracy, and why LP-FT mitigates it. Frame it as the head's
initial gradient direction pulling the features off the pretrained manifold.

**P10 (open: choosing a backbone).** You must classify a 5-class, 250-image, out-of-domain (microscopy) dataset
and care about robustness to staining shifts. Argue for a backbone (supervised ImageNet vs. DINOv2 vs. CLIP vs.
MAE) and a strategy (linear probe / LP-FT / WiSE-FT / PEFT), citing the pretraining objective's effect on frozen
vs. fine-tuned transfer and the divergence/adaptability terms. There is no single right answer; defend yours.

---

## Solution sketches

**S1.** Freeze-transfer accuracy is high for small `k` (general features), then *drops* as `k` grows because (i)
late features are source-specific and (ii) mid-stack splits break co-adaptation. Fine-tuning after transfer stays
high for all `k` because it *heals* both — the co-adaptation is re-established and specificity is re-tuned.

**S2.** (a) `~E*N*C`. (b) `~N*C + E*N*c`. Speedup `= E*N*C / (N*C + E*N*c) = E*C/(C + E*c) -> C/c` as `E->inf`.
Caching pays off increasingly with epochs, approaching the backbone/head cost ratio.

**S3.** Intended activation is `w*((x - mu)/sigma)`; the wrong one is `w*x`, differing by the affine map
`w*mu/... ` — effectively wrong scale and offset per channel. Later layers assume in-distribution activation
statistics, so the mismatch compounds nonlinearly through ReLUs/BN. It is silent because the loss still decreases
(the head compensates somewhat) — the model just plateaus below its true ceiling.

**S4.** (a) small/similar -> linear probe. (b) large/similar -> fine-tune freely. (c) small/different -> hardest:
earlier-layer features + minimal fine-tune, or PEFT (adapters/LoRA) which unlocks the small-data corner. (d)
large/different -> fine-tune extensively, from-scratch competitive. PEFT most changes (c), enabling more
adaptation without overfitting.

**S5.** `d_{H Delta H} approx 2(2a - 1)` (twice the advantage over chance). `a = 0.5`: domains indistinguishable
to `H`, divergence ~0, transfer favorable. `a = 1.0`: perfectly separable, divergence maximal, frozen-`phi`
transfer poor — you must reduce divergence (fine-tune/align) or accept the ceiling.

**S6.** `eps_S` small and `d_{H Delta H}` large points at the divergence term: fine-tune / align features (or use a
more general backbone / earlier layers). The decisive experiment: after alignment, does `eps_T` fall? If it stays
high with divergence near chance, `lambda` is large (concept shift) — you need target labels and real adaptation,
not more alignment. If `lambda` is small, alignment suffices.

**S7.** (a) Covariate: photos vs. sketches of the same object classes — same `p(y|x)` semantics, different `p(x)`;
aligning features works. (b) Concept: a domain where the same texture means "benign" in one imaging modality and
"malignant" in another — `p(y|x)` flips; no fixed head on the shared `phi` can satisfy both, so `lambda` is large
by definition.

**S8.** With independent zero-mean noise, `E[(w_T - w*)^2] = eta^2 s^2 T` (variance accumulates linearly). Large
`eta` early — when the random head produces large `s` — inflates displacement quadratically in `eta`; warmup keeps
`eta` tiny during the noisy phase, and a small LR bounds total drift. Layer-wise decay assigns the smallest `eta`
to early layers, minimizing their displacement most.

**S9.** A random head's first gradients point in an arbitrary direction; back-propagated into `phi` they rotate
the features to fit that bad head, moving them off the pretrained manifold that gave OOD robustness. ID accuracy
recovers (the head catches up) but OOD suffers because the features were distorted. LP-FT first fits a *good* head
(features frozen), so when fine-tuning begins the head gradients are already sensible and barely perturb `phi`.

**S10.** Defensible answer: DINOv2 or CLIP for strong, robust frozen features, with **linear probe or LP-FT** given
only 250 images (small-data corner favours freezing / minimal adaptation); WiSE-FT if you fine-tune and have a
shifted eval, to preserve staining-shift robustness. Microscopy is out-of-domain (large divergence from natural
images), so consider earlier-layer features or PEFT; supervised ImageNet is the weakest choice here (features tied
to natural-object categories). Argue via: SSL/CLIP frozen features are strong and general (lower `eps_S` and,
after light adaptation, lower divergence), and robustness matters, so avoid naive full fine-tune (distortion).
