# Week 3 — Quiz

Ten questions. Answer key below.

**1. A fully-connected layer is a poor fit for images mainly because:**

- A. It explodes in parameters and ignores spatial structure
- B. It only works in grayscale
- C. It is too slow to run
- D. It cannot use a GPU

**2. Weight sharing in a conv layer means:**

- A. The same kernel is reused at every spatial position
- B. The bias is zero
- C. Weights are shared between GPUs
- D. Each pixel has its own weights

**3. A Conv2d with in=3, out=16 produces an output with how many channels?**

- A. 16
- B. 1
- C. 3
- D. 48

**4. Max pooling with a 2×2 window and stride 2:**

- A. Adds channels
- B. Increases resolution
- C. Halves height and width, keeping the strongest response per block
- D. Doubles the spatial size

**5. The receptive field of a neuron is:**

- A. Its number of weights
- B. Its activation value
- C. The region of the original image that influences it
- D. The learning rate

**6. Stacking two 3×3 convolutions gives each neuron a receptive field of:**

- A. 3×3
- B. 1×1
- C. 9×9
- D. 5×5

**7. Data augmentation helps mainly by:**

- A. Speeding up training
- B. Reducing the model size
- C. Generating plausible new training images to reduce overfitting
- D. Removing the need for a test set

**8. Augmentation should be applied to:**

- A. Training and test data equally
- B. Neither
- C. Test data only
- D. Training data only

**9. First-layer CNN filters, once trained, often resemble:**

- A. Full object templates
- B. Edge and color detectors like classical Gabor/Sobel filters
- C. Random noise forever
- D. Text

**10. The best evidence a CNN beats a dense net on images is:**

- A. Higher accuracy on a held-out test set
- B. A prettier loss curve
- C. Lower training loss
- D. Fewer epochs

---

## Answer key

1. **A. It explodes in parameters and ignores spatial structure** — Dense layers need millions of weights and lose the grid; convolution fixes both.
2. **A. The same kernel is reused at every spatial position** — Reusing one kernel everywhere drops parameters and gives translation equivariance.
3. **A. 16** — Each of the 16 kernels produces one output feature map, so out_channels=16.
4. **C. Halves height and width, keeping the strongest response per block** — It downsamples by 2, growing the receptive field and reducing compute.
5. **C. The region of the original image that influences it** — Deeper neurons have larger receptive fields, eventually covering the whole image.
6. **D. 5×5** — Each extra 3×3 conv adds 2 to the field size — the VGG stack-small-convs insight.
7. **C. Generating plausible new training images to reduce overfitting** — Label-preserving transforms expand the effective dataset and regularize.
8. **D. Training data only** — You evaluate on clean images; augmenting the test set would distort the metric.
9. **B. Edge and color detectors like classical Gabor/Sobel filters** — The network rediscovers edge/color detectors that engineers used to hand-design.
10. **A. Higher accuracy on a held-out test set** — Held-out accuracy — not training loss — is what proves real generalization.
