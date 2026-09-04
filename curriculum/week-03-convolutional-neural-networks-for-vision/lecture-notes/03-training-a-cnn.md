# Lecture 3 — Training a CNN that works: normalization and augmentation as regularization

You have the pieces. Now assemble and *train* a CNN, and meet the practical realities that decide
whether it learns anything: the data pipeline, normalization, augmentation, and honest evaluation. The C53
training loop carries over unchanged; what is new is vision-specific regularization, which we treat as
theory rather than folklore.

## The data pipeline

Images need preprocessing before the network:

- **Normalize.** Scale pixels so each channel has mean ~0 and std ~1 using *dataset* statistics. This is not
  cosmetic: un-normalized inputs make the loss Hessian ill-conditioned (recall the condition-number story
  from optimization), so gradient descent crawls or diverges. Normalize with the *training* set's mean/std
  and apply the same constants to val/test — computing statistics on the test set leaks information.
- **Batch** with a `DataLoader` — vision datasets are large, so you stream mini-batches from disk with
  parallel workers.

```python
from torchvision import datasets, transforms
tf = transforms.Compose([
    transforms.ToTensor(),                       # (H,W,C) uint8 -> (C,H,W) float in [0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR-10 stats
])
train = datasets.CIFAR10(root=".", train=True, download=True, transform=tf)
loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
```

## Batch normalization: why it helps

**Batch normalization** (Ioffe & Szegedy, 2015) normalizes each channel's pre-activations across the batch
to zero mean, unit variance, then rescales by learned `gamma, beta`:

    x_hat = (x - mu_B) / sqrt(var_B + eps),   y = gamma * x_hat + beta.

At test time it uses running estimates of `mu, var`. The original paper credited "reduced internal
covariate shift," but Santurkar et al. (2018, "How Does Batch Normalization Help Optimization?") showed the
real mechanism is a **smoother loss landscape** — BN reduces the Lipschitz constant of the loss and its
gradients, permitting higher learning rates and faster, more stable training. Caveat: BN couples examples
in a batch, so it degrades with tiny batches; **Group Normalization** (Wu & He, 2018) is the standard fix.

## Dropout and weight decay

**Dropout** (Srivastava et al., 2014) randomly zeros activations during training, acting like an ensemble
over sub-networks and discouraging co-adaptation; it is used mostly in dense heads now that BN regularizes
conv stacks. **Weight decay** (L2 penalty on weights) shrinks the solution norm, trading a little bias for
variance — the same minimum-norm bias that helps generalization.

## Data augmentation: label-preserving invariance, injected

Images have a superpower text lacks: you can synthesize *plausible new labeled examples* by transforming
existing ones. Random crops, horizontal flips, small rotations, and color jitter teach the network that a
cat is a cat regardless of position, mirror, or lighting. Formally, augmentation is a way to **encode
known invariances of the label** into training — you are telling the model "these transforms do not change
the class." It is among the most effective anti-overfitting tools in vision and essentially free.

```python
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
```

Two rules. **Only augment training data**, never val/test — you evaluate on clean images. And only use
*label-preserving* transforms: a horizontal flip keeps a cat a cat, but a vertical flip or a large rotation
can break labels (a '6' becomes a '9'; a landscape becomes nonsense). Stronger schemes — Cutout (DeVries &
Taylor, 2017), Mixup (Zhang et al., 2018), RandAugment (Cubuk et al., 2020) — push further and often add
a couple of points on CIFAR-10.

## The training loop and honest evaluation

The loop is C53's, unchanged: `zero_grad -> forward -> loss -> backward -> step`, with `CrossEntropyLoss`
(which fuses log-softmax + NLL for numerical stability) and SGD-with-momentum or Adam. What matters for
vision honesty:

- **A held-out validation/test set.** Report accuracy on images never trained on. A CNN can memorize
  CIFAR-10 — training accuracy near 100% with far-lower validation accuracy is overfitting, not skill.
- **Track both curves.** Plot train and validation accuracy together; the gap is your overfitting gauge and
  tells you whether to add capacity (gap small, both low) or regularization (gap large).
- **A confusion matrix, not just accuracy.** See *which* classes it confuses (cats vs. dogs, 4 vs. 9). That
  is where real understanding lives, and it drives targeted fixes.
- **A learning-rate schedule.** A cosine or step decay (Loshchilov & Hutter, 2017) reliably buys accuracy;
  a constant LR leaves points on the table.

## What to expect

A small CNN with augmentation reaches ~99% on MNIST and ~80-90% on CIFAR-10, and crucially *beats a
fully-connected baseline by a wide margin* — proving convolution's structural advantage. If your CNN does
not beat a dense net on images, something is wrong: bad normalization, no augmentation, a shape bug, or a
dead optimizer. Print shapes, check the loss actually decreases on a single batch (overfit one batch to
100% as a sanity test), and verify your data statistics.

**Takeaway:** train a CNN with the C53 loop plus vision essentials — normalize with training-set statistics,
regularize with batch norm and label-preserving augmentation, schedule the learning rate, and evaluate
honestly with both curves and a confusion matrix. BN helps by smoothing the loss landscape, augmentation by
injecting known label invariances; a correctly built CNN must beat a dense baseline, and that gap is
convolution earning its keep.
