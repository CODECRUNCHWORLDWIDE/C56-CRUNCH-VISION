# Week 3 — Graduate Problem Set: Convolution Algebra, Receptive Fields, Backprop, and Architecture

Eleven problems, easy to hard, mixing derivation, proof, computation, and open analysis. Solution
sketches are at the end — attempt each fully before reading them. Notation: a conv layer maps input
`X in R^{C_in x H x W}` to output `Y in R^{C_out x H' x W'}` with kernels `K in R^{C_out x C_in x k x k}`.

**P1 (shape arithmetic).** For `Conv2d(in=16, out=32, k=5, stride=2, padding=2, dilation=1)` on a
`(N, 16, 64, 64)` input, compute `H'` and `W'` from the size formula. Then give the padding needed for a
`k=7, stride=1` conv to preserve spatial size ("same").

**P2 (parameter and FLOP counting).** (a) How many learnable weights (with bias) does the P1 layer have?
(b) Using `C_in*C_out*k*k*H'*W'`, compute its forward multiply-adds. (c) A dense layer producing the same
output from the same flattened input would have how many weights? Give the ratio.

**P3 (equivariance forces convolution).** State what it means for a linear map `y = Wx` on a 1-D signal to
commute with the shift operator `S` (`WS = SW`). Show this forces `W` to be circulant, and explain in one
sentence why circulant = convolution. (You may argue with the first row of `W`.)

**P4 (receptive field recurrence).** Derive the recurrence `r_out = r_in + (k-1)*j_in`, `j_out = j_in*s`
from the definition of the receptive field. Use it to compute the theoretical RF of: (a) five stacked 3x3
stride-1 convs; (b) the same but with a 2x2 stride-2 pool after every conv.

**P5 (small vs. large kernels).** Prove that a stack of two 3x3 stride-1 convs has the same receptive field
as one 5x5 conv, but compare their parameter counts (per channel-pair) and the number of nonlinearities.
Generalize: how many stacked 3x3s match a single (2m+1)x(2m+1) conv, and what is the parameter ratio?

**P6 (kernel gradient).** From `Y[i,j] = sum_{a,b} K[a,b] X[i+a, j+b]` (single channel) and upstream
gradient `dY`, derive `dL/dK[a,b]`. Identify the operation (correlation of what with what).

**P7 (input gradient and the flip).** Derive `dL/dX[p,q]` in terms of `dY` and `K`. Show it equals a *full*
convolution of `dY` with the 180-degree-flipped kernel, and explain precisely why the flip appears (which
index substitution produces it).

**P8 (im2col as a linear operator).** Argue that convolution can be written `y = A x` where `A` is a fixed
doubly-block-Toeplitz matrix. Using this, show that the input-gradient is `A^T dY`, and connect `A^T` to the
col2im scatter-add. Why is `A^T` a *scatter*-add rather than an overwrite?

**P9 (1x1 convolutions).** Show that a `1x1` conv from `C_in` to `C_out` channels applied at every spatial
location is exactly a per-pixel linear map (matrix `C_out x C_in`). Compute its FLOPs and explain why
Inception uses it as a bottleneck before 3x3/5x5 convs.

**P10 (residual gradient flow).** For a residual block `y = F(x; theta) + x`, compute `dy/dx` and, for a
stack of `L` such blocks, express the gradient of the loss w.r.t. the first block's input as a product/sum
involving the `dF/dx` terms and identity. Explain why this prevents the gradient from vanishing even when
each `dF/dx` is small, and contrast with a plain (non-residual) deep stack.

**P11 (open analysis — degradation).** He et al. (2016) observed that a 56-layer *plain* net had higher
*training* error than a 20-layer one, even though the deeper net can represent every function the shallower
one can (set extra layers to identity). Explain this apparent paradox: why can a network *represent* a
solution that its optimizer cannot *find*? Discuss what residual connections change about the optimization
landscape, and where the analogy to identity-initialization breaks down. (Open-ended; argue carefully.)

---

## Solution sketches

**S1.** `H' = floor((64 + 2*2 - 1*(5-1) - 1)/2) + 1 = floor(63/2)+1 = 31+1 = 32`; same for `W'`, so
`(N,32,32,32)`. "Same" for k=7, s=1 needs `p=(k-1)/2=3`.
**S2.** (a) `16*32*5*5 + 32 = 12,832`. (b) `16*32*5*5*32*32 = 13,107,200` MACs. (c) dense: inputs
`16*64*64=65,536`, outputs `32*32*32=32,768` -> `~2.15e9` weights; ratio ~`1.7e5`x more than the conv.
**S3.** `WS=SW` means shifting the input then applying `W` equals applying `W` then shifting. This forces
each row of `W` to be a shifted copy of the previous row (constant along diagonals) = circulant; multiplying
by a circulant matrix is convolution with the kernel given by its first column.
**S4.** From `Y[i]` depending on inputs `i*s .. i*s+(k-1)*j_in` (jump `j_in`), the span grows by `(k-1)*j_in`
and the jump multiplies by `s`. (a) `r=1+2*5=11`; wait per-layer: r:1->3->5->7->9->11, so `11x11`, j=1.
(b) with pool (k=2,s=2) after each conv, j doubles each block; RF grows in strides of 1,2,4,8,16 -> compute
iteratively (r ends much larger, ~`>60`), showing pooling accelerates coverage.
**S5.** Two 3x3s: RF 5x5 (S4), params `2*9=18` per channel-pair vs. `25` for one 5x5, and 2 nonlinearities
vs. 1. `m` stacked 3x3s match a `(2m+1)x(2m+1)` conv; params `9m` vs. `(2m+1)^2`, so small kernels win for
`m >= 1` and add nonlinearity — the VGG argument.
**S6.** `dL/dK[a,b] = sum_{i,j} dY[i,j] * X[i+a,j+b]` = cross-correlation of input `X` with upstream `dY`.
**S7.** `X[p,q]` affects `Y[i,j]` when `(a,b)=(p-i,q-j)` is valid; so `dL/dX[p,q]=sum_{i,j} dY[i,j]
K[p-i,q-j]`. The substitution `a=p-i` flips the kernel index sense, giving a full convolution with
`flip(K)`.
**S8.** Convolution is linear in `X`, so `y = A x` with `A` sparse doubly-block-Toeplitz. Then
`dL/dx = A^T dL/dy = A^T dY`. `A^T` maps each output back to the (overlapping) inputs that produced it;
because inputs feed multiple outputs, `A^T` *accumulates* contributions — a scatter-add (col2im).
**S9.** A 1x1 conv at pixel `(i,j)` computes `Y[:,i,j] = M X[:,i,j]` with `M in R^{C_out x C_in}` — a linear
map per pixel. FLOPs `C_in*C_out*H*W`. As a bottleneck it cheaply reduces `C_in` before an expensive 3x3,
cutting that layer's `C_in*C_out*k*k*...` cost.
**S10.** `dy/dx = I + dF/dx`. Over `L` blocks, backprop multiplies factors `(I + dF_l/dx)`; expanding, the
identity term guarantees a path of magnitude ~1 to the first block regardless of the small `dF/dx` terms, so
gradients do not vanish. A plain stack multiplies `dF_l/dx` factors only, which shrink geometrically.
**S11.** Representability is not reachability: SGD searches a nonconvex landscape and making many stacked
nonlinear layers *learn* the identity is hard (each must undo the previous). Residuals re-parameterize so
that identity = zero residual (easy, and encouraged by weight decay) and give gradients the `+I` highway, so
the optimizer starts near identity and only has to learn *departures* from it. The analogy breaks when the
input/output dimensions differ (needing a projection shortcut) and at extreme depth where even residual
gradients need normalization to stay well-scaled.
