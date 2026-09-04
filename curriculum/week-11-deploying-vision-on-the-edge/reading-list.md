# Week 11 — Reading List

Primary sources for Week 11. Start with the three starred essentials, then read for depth. Prefer these papers and official docs to blog posts — they are where the methods are stated correctly.

- **Howard, Zhu, Chen, Kalenichenko, Wang, Weyand, Andreetto & Adam (2017), 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,' arXiv** — depthwise-separable convolutions and the width multiplier; the foundation of on-device vision. Read for the FLOP arithmetic of Lecture 1.
- **Jacob et al. (2018), 'Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference,' CVPR** — the affine INT8 scheme, integer-only inference, and QAT with fake-quant; the canonical quantization reference. Essential for Lectures 2 and 4.
- **Williams, Waterman & Patterson (2009), 'Roofline: An Insightful Visual Performance Model for Multicore Architectures,' Communications of the ACM** — arithmetic intensity and the compute-vs-memory-bound distinction that governs whether a FLOP cut helps at all.
- Sandler, Howard, Zhu, Zhmoginov & Chen (2018), 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' CVPR — inverted residuals and linear bottlenecks; why the residual connects the thin ends and the narrow layer drops the ReLU.
- Tan & Le (2019), 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,' ICML — compound scaling and the alpha·beta²·gamma²~2 constraint for accuracy per FLOP.
- Hinton, Vinyals & Dean (2015), 'Distilling the Knowledge in a Neural Network,' NeurIPS Deep Learning Workshop — knowledge distillation, temperature, soft targets, and the T²-scaled loss.
- Han, Pool, Tran & Dally (2015), 'Learning both Weights and Connections for Efficient Neural Networks,' NeurIPS — magnitude pruning and iterative prune-then-fine-tune; the basis for structured/unstructured pruning.
- Bengio, Léonard & Courville (2013), 'Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation,' arXiv — the straight-through estimator that makes QAT differentiable through rounding.
- Krishnamoorthi (2018), 'Quantizing deep convolutional networks for efficient inference: A whitepaper,' arXiv — a practical, thorough guide to PTQ vs. QAT, per-channel quantization, and calibration; the best single how-to.
- Tan, Chen, Pang, Vasudevan, Sandler, Howard & Le (2019), 'MnasNet: Platform-Aware Neural Architecture Search for Mobile,' CVPR — hardware-aware NAS optimizing *measured* on-device latency, because FLOPs mispredict speed.
- Lin, Chen, Lin, Gan & Han (2020), 'MCUNet: Tiny Deep Learning on IoT Devices,' NeurIPS — co-designing network and runtime to fit ImageNet-scale vision into ~320 KB of microcontroller SRAM; peak activation memory as the binding constraint.
- Chen et al. (2018), 'TVM: An Automated End-to-End Optimizing Compiler for Deep Learning,' OSDI — autotuning kernels per operator per target; why the same graph can run many-x apart depending on the compiler.
- ONNX Runtime and PyTorch Quantization documentation — the authoritative reference for export, opsets, graph optimization, and the quantization APIs used in the exercises.
