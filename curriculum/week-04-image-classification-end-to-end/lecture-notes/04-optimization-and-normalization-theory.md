# Lecture 4 — Why the optimizer works: SGD noise, momentum, AdamW, and what batch norm really does

Lecture 3 gave you recipes — SGD+momentum, AdamW, batch norm, cosine schedules. This lecture
derives *why* they work, so you can predict their behavior instead of memorizing defaults. The four ideas are:
SGD is gradient descent with a specific noise, momentum is a heavy-ball accelerator, AdamW is a per-coordinate
preconditioner, and batch normalization reshapes the loss landscape rather than merely "reducing internal
covariate shift."

## Stochastic gradient descent as noisy descent

Full-batch gradient descent uses `∇R̂(θ) = (1/n) Σ_i ∇ℓ_i(θ)`. SGD replaces it with a mini-batch estimate
`g_B(θ) = (1/|B|) Σ_{i∈B} ∇ℓ_i(θ)`. Crucially `E_B[g_B] = ∇R̂`, so the mini-batch gradient is an **unbiased
estimator** of the full gradient, with covariance that scales like `1/|B|`. The update

    θ_{t+1} = θ_t − η g_B(θ_t) = θ_t − η ∇R̂(θ_t) + η ξ_t,   E[ξ_t] = 0,

is therefore gradient descent plus zero-mean noise of scale `η/sqrt(|B|)`. Two consequences follow. First, the
step-size ceiling from Lecture 3 (`η < 2/λ_max`) still bounds the deterministic part; a batch too small or an
LR too large lets the noise term dominate and the loss becomes jagged or diverges. Second, the noise is not
purely harmful: it lets the iterate escape sharp/narrow minima and settle in *flatter* ones, which empirically
generalize better (Keskar et al., 2017, "On Large-Batch Training..."; Jastrzębski et al., 2017). This is the
theoretical reason the **effective learning rate is `η/|B|`** and why scaling batch size usually requires
scaling LR (Goyal et al., 2017, linear scaling rule with warmup). SGD's noise is a feature.

## Momentum: the heavy-ball accelerator

Momentum maintains a velocity and updates

    v_{t+1} = μ v_t + g_B(θ_t),      θ_{t+1} = θ_t − η v_{t+1}.

On a quadratic with condition number `κ = λ_max/λ_min`, plain GD needs `O(κ)` steps; the heavy-ball / Nesterov
method needs `O(sqrt(κ))` — a quadratic speedup on ill-conditioned problems (Polyak, 1964; Nesterov, 1983).
Intuitively, `v` averages recent gradients: across a steep, oscillating direction the sign flips and the
average cancels (damping the zig-zag), while along a consistent shallow direction the contributions accumulate
(accelerating progress). The trade is that too large a `μ` (e.g. 0.99) overshoots; `μ = 0.9` is the standard
vision default. Momentum is why SGD-with-momentum reaches good CNN solutions in far fewer epochs than vanilla
SGD.

## Adaptive methods: AdamW as a diagonal preconditioner

Adam keeps exponential moving averages of the gradient (`m_t`, first moment) and its square (`v_t`, second
moment) and steps

    θ_{t+1} = θ_t − η · m̂_t / (sqrt(v̂_t) + ε),

with bias-corrected `m̂, v̂`. Dividing by `sqrt(v̂_t)` gives each coordinate its *own* effective step: parameters
with large, noisy gradients take smaller steps; rarely-updated parameters take larger ones. This is a cheap
diagonal approximation to preconditioning by the curvature, which is why Adam is robust to bad scaling and
converges fast with little tuning — at some cost in final generalization on CNNs versus well-tuned SGD (Wilson
et al., 2017). The **W** matters: in plain Adam, adding L2 to the loss makes the weight-decay term flow through
the adaptive denominator, so it is effectively scaled per-coordinate and no longer a clean penalty. **AdamW**
(Loshchilov & Hutter, 2019) *decouples* it, applying `θ ← θ − η λ θ` directly. This single fix restored weight
decay's regularizing effect and made AdamW the default for Transformers. If you use Adam with `weight_decay`,
make sure it is AdamW.

## Batch normalization: what it actually normalizes

BatchNorm normalizes each feature over the mini-batch, then rescales:

    x̂ = (x − μ_B) / sqrt(σ_B² + ε),      y = γ x̂ + β,

with learned `γ, β`, and running estimates of `μ, σ` used at inference. The original paper (Ioffe & Szegedy,
2015) motivated it as reducing "internal covariate shift," but Santurkar et al. (2018, "How Does Batch
Normalization Help Optimization?") showed experimentally that the covariate-shift story is not the mechanism.
The real effect is that BN makes the **loss landscape smoother** — it reduces the Lipschitz constant of the loss
and of its gradients — so gradients are more predictive over longer distances, which *permits larger learning
rates and faster, more stable training*. Practical consequences you must respect: BN couples examples within a
batch, so it behaves badly at very small batch sizes (use GroupNorm/LayerNorm there); its train/eval behavior
differs (running stats), so forgetting `model.eval()` at test time silently corrupts predictions; and it
interacts with weight decay in subtle ways because scaling weights before BN is a no-op for the forward pass.

## Worked micro-example: why `η/|B|` is the effective rate

Suppose per-example gradients have variance `s²` per coordinate. A batch of size `|B|` has gradient-estimate
variance `s²/|B|`, so one SGD step injects noise of scale `η · s/sqrt(|B|)`. Doubling the batch halves the noise
std (`1/sqrt2`); to keep the same exploration you roughly double `η` — the linear scaling rule — but only up to
the point where the deterministic step `η` nears the `2/λ_max` ceiling, beyond which warmup is required to avoid
early divergence. This back-of-envelope predicts real large-batch training behavior.

## Common pitfalls

- **Plain Adam with `weight_decay`.** The penalty is distorted by the adaptive denominator; use AdamW.
- **BatchNorm at batch size 1–2.** Batch statistics are meaningless; switch to GroupNorm/LayerNorm.
- **Forgetting `model.eval()`.** BN and dropout change behavior between train and eval; the classic "great
  training, broken test" bug.

**Takeaway:** SGD is unbiased noisy gradient descent whose noise scale `η/sqrt(|B|)` both bounds stability and
helps find flat, generalizing minima; momentum is a heavy-ball accelerator turning `O(κ)` into `O(sqrt(κ))`;
AdamW is a decoupled-decay diagonal preconditioner; and batch norm helps by smoothing the loss landscape, not
by fixing covariate shift. Knowing the mechanism lets you predict, not guess, how each choice behaves.
