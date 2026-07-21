# Mini-Project — A CNN Image Classifier, Honestly Evaluated

## Brief

Build, train, and honestly evaluate a convolutional image classifier, proving you understand why CNNs beat dense nets on images and how to train one that generalizes.

## Requirements

1. **Data:** MNIST or CIFAR-10 via torchvision, normalized, in train/test DataLoaders, with label-preserving augmentation on training data only.
2. **Model:** a CNN of stacked conv+ReLU+pool (or strided conv) blocks ending in a classifier head. Channels grow, spatial size shrinks.
3. **Baseline:** a fully-connected network of comparable size, to quantify convolution's advantage.
4. **Training:** the C53 loop with CrossEntropyLoss and a sensible optimizer; log and plot train + validation accuracy.
5. **Evaluation:** final test accuracy, a confusion matrix, and an error analysis naming the most-confused classes — with a few misclassified images shown.
6. **Interpretability:** visualize the first conv layer's learned filters.

## Stretch

- Add batch normalization and/or dropout and measure their effect on the overfitting gap.
- Push CIFAR-10 accuracy with a deeper architecture and a learning-rate schedule.

## What you're proving

You can design, train, and *evaluate* a CNN — and look inside it. You've now closed the loop from Week 1's hand-picked kernels to a network that learns its own. Next week you scale this to a full, production-shaped classification pipeline; the week after, you stop training from scratch and stand on pretrained giants.
