# Mini-Project — A CNN Image Classifier, Honestly Evaluated and Fully Understood

## Brief

Build, train, and honestly evaluate a convolutional image classifier that beats a dense baseline, and prove
you understand the operator underneath it by shipping a from-scratch, gradient-checked convolution. This is
the deliverable that closes the loop from Week 1's hand-picked kernels to a network that learns its own — and
that knows *why* it works.

## Requirements

1. **Data.** MNIST or CIFAR-10 via torchvision, normalized with **training-set channel statistics**, in
   train/test DataLoaders, with label-preserving augmentation (random crop + horizontal flip) on training
   data only.
2. **Model.** A CNN of stacked conv->BatchNorm->ReLU->pool (or strided conv) blocks ending in **global
   average pooling** and a linear head. Channels grow, spatial size shrinks. Print the shape after each block
   once to confirm your arithmetic.
3. **Baseline.** A fully-connected network of *comparable parameter count*, to quantify convolution's
   advantage. Report both parameter counts.
4. **Training.** The C53 loop with CrossEntropyLoss, SGD-with-momentum + weight decay, and a cosine LR
   schedule. Do the single-batch overfit sanity check first. Log and plot train + validation accuracy.
5. **Evaluation.** Final held-out test accuracy, a confusion matrix, and an error analysis naming the
   most-confused classes with a few misclassified images shown.
6. **Interpretability.** Visualize the first conv layer's learned filters and comment on what they resemble.
7. **From-scratch operator.** Implement a convolution forward *and* backward pass from scratch and pass a
   central-difference **numerical gradient check** (relative error < 1e-5) *and* a cross-check against
   `torch.autograd`. Include one sentence on why the input-gradient needs the kernel flip.

## Stretch

- Add a residual (skip) connection to a deeper variant and show it improves *training* accuracy — a
  small-scale degradation experiment.
- Ablate batch norm and augmentation independently and report each one's contribution with a results table.
- Push CIFAR-10 accuracy with a deeper residual architecture and stronger augmentation (Cutout/Mixup).

## What you are proving

You can design, train, and *evaluate* a CNN, look inside it, and — crucially — implement and validate the
convolution operator that everything else rests on. Next week you scale this into a full, production-shaped
classification pipeline; the week after, you stop training from scratch and stand on pretrained giants.
