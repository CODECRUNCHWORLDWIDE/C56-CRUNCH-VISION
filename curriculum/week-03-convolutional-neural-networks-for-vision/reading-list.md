# Week 3 — Reading List

Primary sources for Week 3. Start with the three starred essentials, then read for depth. These are the papers where the ideas are stated correctly — prefer them to secondary summaries.

- **LeCun, Bottou, Bengio & Haffner (1998), 'Gradient-based learning applied to document recognition,' Proc. IEEE** — LeNet-5; the paper that established the conv-extractor-then-head template. The origin of everything this week.
- **Krizhevsky, Sutskever & Hinton (2012), 'ImageNet Classification with Deep CNNs,' NeurIPS (AlexNet)** — the result that ignited modern deep learning: ReLU, dropout, augmentation, GPUs, at ImageNet scale.
- **He, Zhang, Ren & Sun (2016), 'Deep Residual Learning for Image Recognition,' CVPR (ResNet)** — the degradation problem and residual connections; the most important architectural idea in the lecture, and it recurs in Week 10.
- Simonyan & Zisserman (2015), 'Very Deep Convolutional Networks for Large-Scale Image Recognition,' ICLR (VGG) — the stacked-3x3 receptive-field argument, made systematic.
- Szegedy et al. (2015), 'Going Deeper with Convolutions,' CVPR (GoogLeNet/Inception) — 1x1 bottlenecks and global average pooling for efficiency; the FLOP-conscious counterpoint to VGG.
- Ioffe & Szegedy (2015), 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,' ICML — read together with Santurkar et al. (2018), 'How Does Batch Normalization Help Optimization?', NeurIPS, which corrects the mechanism to loss-landscape smoothing.
- Luo, Li, Urtasun & Zemel (2016), 'Understanding the Effective Receptive Field in Deep CNNs,' NeurIPS — proves the effective RF is Gaussian and O(sqrt(depth)); the surprising fact behind Lecture 2.
- Zeiler & Fergus (2014), 'Visualizing and Understanding Convolutional Networks,' ECCV — deconvnet feature visualization; the basis for Challenge 1 and for what feature maps mean.
- Lin, Chen & Yan (2014), 'Network in Network,' ICLR — introduces 1x1 convolutions and global average pooling, both structural staples used all week.
- Yu & Koltun (2016), 'Multi-Scale Context Aggregation by Dilated Convolutions,' ICLR — dilated/atrous convolution for enlarging the receptive field cheaply; sets up segmentation in Week 7.
- Goodfellow, Bengio & Courville, Deep Learning (MIT Press, 2016), Ch. 9 ('Convolutional Networks') — the canonical graduate textbook treatment of convolution, pooling, and the backward pass.
- Stanford CS231n, 'Convolutional Neural Networks' course notes — the authoritative, careful lecture notes on conv arithmetic, receptive fields, and layer patterns.
