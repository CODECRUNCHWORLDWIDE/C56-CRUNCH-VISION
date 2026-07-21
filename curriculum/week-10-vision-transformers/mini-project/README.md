# Mini-Project — Vision Transformer, Built and Benchmarked

## Brief

Understand and benchmark the architecture reshaping vision: build the ViT's front door yourself, run a real one, and compare it honestly to a CNN — proving you can reason about ViTs vs. CNNs with measurement, not fashion.

## Requirements

1. **Patch embedding from scratch:** turn an image into a `(N, 197, dim)` token sequence (patches + positional embedding + [CLS]); confirm the conv-equals-patch-embed equivalence and visualize the patch grid.
2. **Run a pretrained ViT:** predictions on your images with correct preprocessing; visualize an attention map if accessible.
3. **ViT vs. CNN:** on a small dataset, compare a fine-tuned ViT and CNN (accuracy, time, overfitting), and ideally the from-scratch collapse that shows the ViT's data-hunger.
4. **Analysis:** a measured, hype-free write-up — for this data and budget, which wins and why, referencing inductive biases, data scale, and the N² cost.
5. **README:** reproduce steps and honest limitations.

## Stretch

- A long-range-relationship task where global attention beats local convolution (Challenge 1).
- A matched ViT vs. ResNet vs. ConvNeXt comparison isolating architecture from training recipe (Challenge 2).

## What you're proving

You understand the Transformer applied to vision at the level of having built its input pipeline, and you can judge ViT vs. CNN by evidence rather than hype. Next week you make vision *practical*: shrinking and deploying a model to run in real time on constrained hardware.
