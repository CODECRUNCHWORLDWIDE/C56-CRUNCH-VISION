# Week 3 — Homework

Cement the operator and the architecture before scaling into a full pipeline next week. Do the derivations by hand before coding — the convolution backward pass and the receptive-field recurrence must be muscle memory before a library hides them. A few focused hours.

## Tasks

- Derive, in writing, why weight sharing follows from demanding translation equivariance, and separately show that a conv layer's parameter count is independent of image size.
- Do the receptive-field recurrence by hand for a 6-layer stack you design (mixing convs, strides, and one pool), then verify a few RF values empirically by zeroing an input pixel and checking which outputs change.
- Derive all three convolution gradients (kernel, input, bias) from the definition, state which is the flipped-kernel full convolution, and note the numerical-check tolerance you would require.
- Add batch normalization to your mini-project CNN and report the change in training stability, usable learning rate, and final accuracy; connect the result to the loss-smoothing explanation.
- Add a residual connection to your deepest CNN variant and measure its effect on *training* accuracy, tying the observation to the dy/dx = dF/dx + I argument.
- Read the CS231n Convolutional Networks notes and He et al. (2016), and write a paragraph distinguishing the *degradation* problem from *overfitting* using their evidence.

## Definition of done

A committed CNN classifier on MNIST or CIFAR-10 that beats a fully-connected baseline of comparable size, trained with normalization, augmentation, batch norm, and a LR schedule, reporting the single-batch sanity check, train/validation curves, a confusion matrix with error analysis, final held-out accuracy, and a visualization of first-layer learned filters — plus a from-scratch convolution forward/backward pass that passes a central-difference gradient check (<1e-5) and matches autograd, with a note on why the input-gradient uses the flipped kernel.

Submit by committing your work to your course repo under `week-03/`.
