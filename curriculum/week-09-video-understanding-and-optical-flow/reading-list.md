# Week 9 — Reading List

Primary sources for Week 9 — flow, video architectures, and self-supervised video. Start with the three starred essentials, then read for depth. Prefer these to blog posts: they are where the ideas are stated (and benchmarked) correctly.

- **Horn & Schunck (1981), 'Determining optical flow,' *Artificial Intelligence*** — the variational formulation, the brightness-constancy constraint, and the smoothness prior; the origin of dense flow and required reading for Lecture 1.
- **Teed & Deng (2020), 'RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,' ECCV (best paper)** — the modern flow standard: all-pairs correlation volume + recurrent GRU update; the core of Lecture 4.
- **Bertasius, Wang & Torresani (2021), 'Is Space-Time Attention All You Need for Video Understanding?' (TimeSformer), ICML** — divided space-time attention and its cost/accuracy analysis; the anchor for Lecture 5.
- Lucas & Kanade (1981), 'An iterative image registration technique...,' IJCAI — the local-constancy least-squares method underlying sparse flow and feature tracking (the structure tensor).
- Simonyan & Zisserman (2014), 'Two-Stream Convolutional Networks for Action Recognition in Videos,' NeurIPS — RGB + optical-flow streams; the breakthrough of feeding explicit motion.
- Carreira & Zisserman (2017), 'Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset' (I3D), CVPR — inflating 2D ImageNet kernels to 3D, and the Kinetics benchmark that shaped the field.
- Tran, Wang, Torresani, Ray, LeCun & Paluri (2018), 'A Closer Look at Spatiotemporal Convolutions for Action Recognition' (R(2+1)D), CVPR — the factorized (2+1)D convolution and why it beats full 3D at lower cost.
- Feichtenhofer, Fan, Malik & He (2019), 'SlowFast Networks for Video Recognition,' ICCV — two pathways at different frame rates; a strong non-Transformer architecture and a lesson in temporal sampling.
- Sun, Yang, Liu & Kautz (2018), 'PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume,' CVPR — the pyramid/warping/cost-volume design bridging classical and learned flow (Lecture 4).
- Tong, Song, Wang & Wang (2022), 'VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training,' NeurIPS — very-high-ratio masking and why video redundancy demands it (Lecture 5).
- Butler, Wulff, Stanley & Black (2012), 'A naturalistic open source movie for optical flow evaluation' (MPI Sintel), ECCV — the benchmark and color-coding convention for evaluating flow.
- Karpathy, Toderici, Shetty, Leung, Sukthankar & Fei-Fei (2014), 'Large-scale Video Classification with Convolutional Neural Networks' (Sports-1M), CVPR — the humbling result that single-frame nearly matches fusion, motivating the whole ladder.
