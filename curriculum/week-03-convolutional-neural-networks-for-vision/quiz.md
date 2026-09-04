# Week 3 — Quiz

Fifteen questions spanning equivariance, receptive-field arithmetic, the convolution backward pass, regularization theory, and the architectural lineage. Attempt each before the answer key.

**1. The deepest reason a fully-connected layer is a poor fit for images is that it:**

- A. only works on grayscale images
- B. requires the image to be square
- C. destroys the spatial grid and is not translation-equivariant, so it must relearn each feature at every location
- D. cannot be run on a GPU

<details>
<summary>Answer</summary>

**C. destroys the spatial grid and is not translation-equivariant, so it must relearn each feature at every location** — Beyond parameter count, flattening discards spatial structure and equivariance; convolution's structural prior is the real win.

</details>

**2. A linear map on a grid is translation-equivariant if and only if it is:**

- A. a diagonal matrix
- B. a permutation
- C. an orthogonal matrix
- D. a convolution (its matrix is circulant)

<details>
<summary>Answer</summary>

**D. a convolution (its matrix is circulant)** — Equivariance forces the weight matrix to be circulant, whose matrix form is exactly convolution — so weight sharing is a theorem.

</details>

**3. For input size W with kernel k, padding p, stride s, dilation d, the output size is:**

- A. (W - k)/s
- B. floor(W/k) + p
- C. floor((W + 2p - d*(k-1) - 1)/s) + 1
- D. W*s + 2p

<details>
<summary>Answer</summary>

**C. floor((W + 2p - d*(k-1) - 1)/s) + 1** — This is the general conv size formula; dilation inflates the effective kernel footprint to d*(k-1)+1.

</details>

**4. Using the recurrence r += (k-1)*j, j *= s, the theoretical receptive field of two stacked 3x3 stride-1 convs is:**

- A. 6x6
- B. 9x9
- C. 3x3
- D. 5x5

<details>
<summary>Answer</summary>

**D. 5x5** — Layer 1: r=3, j=1; layer 2: r=3+2=5. Two 3x3s see 5x5 with fewer weights than one 5x5 plus an extra nonlinearity (VGG).

</details>

**5. The *effective* receptive field of a deep CNN (Luo et al., 2016) is:**

- A. roughly Gaussian, centered, and grows only like O(sqrt(depth)) — smaller than the theoretical RF
- B. uniform over the theoretical RF
- C. exactly equal to the theoretical receptive field
- D. always the entire image after two layers

<details>
<summary>Answer</summary>

**A. roughly Gaussian, centered, and grows only like O(sqrt(depth)) — smaller than the theoretical RF** — Central pixels have more paths to the output, so influence concentrates centrally and the usable field is sublinear in depth.

</details>

**6. The gradient of a convolution with respect to its *input* is:**

- A. a full convolution of the upstream gradient with the 180-degree-flipped kernel
- B. the element-wise product of input and kernel
- C. the spatial sum of the upstream gradient
- D. a correlation of the input with the kernel

<details>
<summary>Answer</summary>

**A. a full convolution of the upstream gradient with the 180-degree-flipped kernel** — dL/dX[p,q] = sum dY[i,j] K[p-i,q-j] — the flipped kernel makes this a full convolution; the flip is load-bearing in backprop.

</details>

**7. The gradient of a convolution with respect to its *kernel* is:**

- A. the cross-correlation of the input X with the upstream gradient dY
- B. the sum of dY over channels
- C. a convolution of dY with the flipped input
- D. the Hessian of the loss

<details>
<summary>Answer</summary>

**A. the cross-correlation of the input X with the upstream gradient dY** — dL/dK[a,b] = sum_{i,j} dY[i,j] X[i+a,j+b], which is exactly the correlation of X with dY.

</details>

**8. im2col implements convolution as a single matrix multiply by:**

- A. unfolding each receptive-field patch into a column so Y = W @ X_col
- B. replacing the kernel with its inverse
- C. sorting the pixels by intensity
- D. taking the FFT of the whole image

<details>
<summary>Answer</summary>

**A. unfolding each receptive-field patch into a column so Y = W @ X_col** — im2col materializes overlapping patches as columns; the reshaped kernels times X_col is the conv, and col2im (scatter-add) is its transpose.

</details>

**9. The forward multiply-add cost of a conv layer is approximately:**

- A. C_out + C_in
- B. H' * W'
- C. C_in * C_out * k * k * H' * W'
- D. k * k

<details>
<summary>Answer</summary>

**C. C_in * C_out * k * k * H' * W'** — Cost is linear in both channel counts and quadratic in kernel size — which is why 1x1 convs (channel-mixing) are so cheap.

</details>

**10. Batch normalization is now understood to help training chiefly by:**

- A. adding trainable parameters that memorize the data
- B. eliminating all overfitting
- C. smoothing the loss landscape (reducing its Lipschitz constant), permitting higher learning rates
- D. removing the need for a validation set

<details>
<summary>Answer</summary>

**C. smoothing the loss landscape (reducing its Lipschitz constant), permitting higher learning rates** — Santurkar et al. (2018) showed the mechanism is a smoother landscape, not the originally-claimed reduction of internal covariate shift.

</details>

**11. Data augmentation reduces overfitting primarily because it:**

- A. shrinks the model's parameter count
- B. increases the learning rate automatically
- C. removes the need for a test set
- D. injects known label-preserving invariances by synthesizing plausible new labeled examples

<details>
<summary>Answer</summary>

**D. injects known label-preserving invariances by synthesizing plausible new labeled examples** — Augmentation encodes invariances of the label (position, mirror, lighting) into training — essentially free regularization.

</details>

**12. Augmentation should be applied to:**

- A. training and test data equally
- B. neither
- C. training data only
- D. test data only

<details>
<summary>Answer</summary>

**C. training data only** — You evaluate on clean images; augmenting val/test would distort the metric and leak the transform into evaluation.

</details>

**13. The 'degradation problem' that motivated ResNet was that very deep plain networks had:**

- A. too few parameters to fit the data
- B. vanishing memory on the GPU
- C. higher test error only, due to overfitting
- D. higher *training* error than shallower ones — an optimization failure, not overfitting

<details>
<summary>Answer</summary>

**D. higher *training* error than shallower ones — an optimization failure, not overfitting** — A 56-layer plain net trained worse than a 20-layer one (He et al., 2016); a deeper net can represent identity but the optimizer could not find it.

</details>

**14. A residual block y = F(x) + x makes very deep networks trainable because:**

- A. the identity is trivial to represent (drive F to 0) and the skip gives gradients a +I highway (dy/dx = dF/dx + I)
- B. it removes all nonlinearities
- C. it doubles the number of parameters
- D. it eliminates the need for normalization

<details>
<summary>Answer</summary>

**A. the identity is trivial to represent (drive F to 0) and the skip gives gradients a +I highway (dy/dx = dF/dx + I)** — The skip makes identity easy to learn so depth never hurts the reachable set, and the +I term stops gradients vanishing.

</details>

**15. GoogLeNet/Inception reduced compute versus VGG mainly by using:**

- A. removing all pooling layers
- B. training without backpropagation
- C. 1x1 convolutions as bottlenecks to shrink channel counts before expensive 3x3/5x5 convs, plus global average pooling
- D. larger 11x11 kernels throughout

<details>
<summary>Answer</summary>

**C. 1x1 convolutions as bottlenecks to shrink channel counts before expensive 3x3/5x5 convs, plus global average pooling** — 1x1 bottlenecks cut the C_in*C_out FLOP term, and global average pooling removed VGG's huge dense head — ~12x fewer parameters.

</details>

---
