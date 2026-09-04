# Week 11 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- Howard et al. — *MobileNets* (v1) and Sandler et al. — *MobileNetV2*: depthwise-separable convs and inverted residuals, the workhorses of on-device vision.
- Tan & Le (2019) — *EfficientNet*: compound scaling for the best accuracy per FLOP.
- [PyTorch — Quantization documentation](https://pytorch.org/docs/stable/quantization.html) and the model-optimization tutorials (dynamic/static PTQ and QAT APIs).
- [ONNX & ONNX Runtime documentation](https://onnxruntime.ai/) — export, opsets, graph optimization, and portable inference; pair with Lecture 3.
- Krishnamoorthi (2018) — *Quantizing deep convolutional networks: a whitepaper*: the best practical guide to PTQ vs. QAT, per-channel quantization, and calibration.
- Hinton, Vinyals & Dean (2015) — *Distilling the Knowledge in a Neural Network*: the distillation idea, temperature, and soft targets.
- [TensorFlow Lite / TFLite for Microcontrollers](https://www.tensorflow.org/lite) — the standard on-device runtime for Android and MCUs; the TinyML entry point (Lecture 5).
- See also [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) for serving, production monitoring, and distribution-shift detection of deployed vision models.

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
