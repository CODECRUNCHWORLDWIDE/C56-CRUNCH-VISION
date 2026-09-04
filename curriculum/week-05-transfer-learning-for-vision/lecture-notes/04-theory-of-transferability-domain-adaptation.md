# Lecture 4 — A theory of transferability: the domain-adaptation bound

So far transferability has been intuition ("features are general") plus empirics (Yosinski, Zeiler &
Fergus). This lecture gives it a theorem. We want to know: if I train on a source distribution `D_S` and deploy
on a target `D_T`, what controls my target error? The answer — the domain-adaptation bound of Ben-David,
Blitzer, Crammer, Kulesza, Pereira & Vaughan (2010, "A theory of learning from different domains", *Machine
Learning*) — decomposes target error into three terms, and every practical transfer trick maps onto shrinking
one of them.

## Setup and the quantity we bound

Fix a hypothesis class `H` (say linear classifiers on top of a fixed representation `phi`). For a hypothesis
`h`, let `eps_S(h)` and `eps_T(h)` be its expected errors on source and target. We can measure and minimize
`eps_S(h)` (we have labeled source data); we care about `eps_T(h)` (we may have little or no target label). We
need to bound the gap.

## The H-divergence

The key object is a divergence between distributions that only "sees" differences a classifier in `H` could
exploit. The **H-divergence** is

    d_H(D_S, D_T) = 2 sup_{h in H} | Pr_{x~D_S}[h(x)=1] - Pr_{x~D_T}[h(x)=1] |.

Read it: if some classifier in `H` can reliably tell source samples from target samples, the domains are far
apart *in a way that matters for `H`*; if no classifier in `H` can distinguish them, they are close. A refined
version, the **H-DeltaH-divergence** `d_{H Delta H}`, uses symmetric differences of hypotheses and is the one in
the bound. Crucially, `d_{H Delta H}` is **estimable from unlabeled data**: train a classifier to discriminate
source from target features; its accuracy above chance *is* (an estimate of) the divergence. This is exactly
what a **domain-adversarial** network (Ganin et al., 2016, DANN) drives to zero.

## The bound

For every `h in H`,

    eps_T(h)  <=  eps_S(h)  +  (1/2) d_{H Delta H}(D_S, D_T)  +  lambda,

where `lambda = min_{h* in H} [ eps_S(h*) + eps_T(h*) ]` is the error of the *best joint hypothesis* — the
**adaptability** term, measuring whether a single `h in H` can do well on both domains at once. (Ben-David et
al., 2010, Theorem 2, with finite-sample versions replacing the true divergence by its empirical estimate plus
a VC-complexity term of order `sqrt(d_VC / m)`.)

## Reading the three terms as engineering levers

This is why the theorem is useful: each term is something you already manipulate.

- **`eps_S(h)` — source error.** Shrunk by using a *better pretrained backbone* and a well-fit head. This is why
  Kornblith et al. (2019) found better ImageNet models transfer better: lower source error is the first term.
- **`d_{H Delta H}` — feature divergence.** Shrunk by making source and target *look the same in feature space*.
  Every domain-alignment method targets this term: fine-tuning moves `phi` so target features resemble source;
  domain-adversarial training (DANN) explicitly minimizes an estimate of the divergence; matching low-level
  statistics (or extracting from earlier, more domain-invariant layers — Lecture 1) reduces it. When you
  "fine-tune more on an exotic domain," you are paying down this term.
- **`lambda` — adaptability.** The *ceiling*. If no classifier on your representation can be simultaneously good
  on both domains — because the label relationship itself changed (conditional shift) — then no amount of
  feature alignment saves you. `lambda` large is the formal statement of "the domains are fundamentally
  incompatible; transfer cannot work here." This is the term the previous lectures gestured at with "transfer
  breaks down on very different domains."

## Covariate shift vs. concept shift

The bound clarifies a distinction you must keep straight. **Covariate shift**: `p(x)` changes but `p(y|x)` is
stable (e.g. photos vs. sketches of the same classes). Here `lambda` is small and aligning features
(`d_{H Delta H}` down) genuinely fixes transfer — the optimistic case. **Concept shift**: `p(y|x)` itself
changes (the same pixels mean different labels across domains). Here `lambda` is large *by construction* and
feature alignment is futile; you need target labels and real adaptation, not just a frozen backbone. Diagnosing
which regime you are in — often by whether a domain classifier on features is separable while a task classifier
transfers poorly — tells you whether more alignment will help or is hopeless.

## Worked micro-example

Suppose `H` = linear heads on a fixed `phi`. You train a domain classifier to separate source (ImageNet-style)
from target (satellite) features and it reaches 95% accuracy: the features are highly separable, so
`d_{H Delta H}` is large — transfer will be poor with the frozen `phi`. You then fine-tune `phi` and re-train the
domain classifier; now it reaches only 60% (near chance for balanced classes is 50%): you have shrunk the
divergence, and — provided `lambda` (checked by whether *any* head does well on both) stayed small — the bound
predicts, and you should observe, a drop in target error. If instead target error stays high while the domain
classifier is at chance, your `lambda` is large: concept shift, and you need target supervision.

## Pitfalls

- **Treating the bound as tight.** It is an upper bound with a loose constant and a VC term; use it to reason
  about *which lever to pull*, not to predict an exact number.
- **Confusing "features look aligned" with "task solved."** Small `d_{H Delta H}` helps only if `lambda` is also
  small. Alignment without adaptability is a mirage.
- **Ignoring label shift.** The bound assumes shared labeling to first order; pure class-prior shift needs its
  own correction (re-weighting), a related but distinct problem.

**Takeaway:** target error is bounded by source error + (half) the H-DeltaH feature divergence + the
best-joint-hypothesis adaptability `lambda` (Ben-David et al., 2010). Better backbones cut the first term,
feature alignment/fine-tuning/DANN cut the second, and `lambda` is the ceiling that decides whether transfer is
even possible — small under covariate shift (alignment works), large under concept shift (it cannot). Every
heuristic in Lectures 1-3 is one of these three levers with a name.
