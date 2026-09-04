# Week 8 — Reading List

Primary sources for Week 8. Start with the three starred essentials, then read for depth. These are the papers that defined heatmap pose, tracking-by-detection, and the modern metric and tracker families — prefer them to blog summaries.

- **Newell, Yang & Deng (2016), 'Stacked Hourglass Networks for Human Pose Estimation,' ECCV** — the canonical heatmap-regression pose architecture; read for the encoder-decoder heatmap head and repeated bottom-up/top-down refinement.
- **Bewley, Ge, Ott, Ramos & Upcroft (2016), 'Simple Online and Realtime Tracking (SORT),' ICIP** — the minimal Kalman + IoU + Hungarian tracker; the baseline every later method is measured against.
- **Luiten, Osep, Dendorfer, Torr, Geiger, Leal-Taixe & Leibe (2021), 'HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking,' IJCV** — the modern standard metric; read for the DetA/AssA decomposition and why MOTA and IDF1 each mislead.
- Wojke, Bewley & Paulus (2017), 'Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT),' ICIP — appearance embeddings + Mahalanobis gating + matching cascade; the SORT-to-Deep-SORT upgrade in full.
- Kalman (1960), 'A New Approach to Linear Filtering and Prediction Problems,' *Journal of Basic Engineering* — the original derivation of the optimal linear-Gaussian estimator underlying every classic tracker.
- Kuhn (1955), 'The Hungarian Method for the Assignment Problem,' *Naval Research Logistics Quarterly* — the O(n^3) optimal assignment algorithm; pair with any LP-duality treatment for the totally-unimodular argument.
- Zhang, Sun, Jiang, Yu, Weng, Yuan, Luo, Liu & Wang (2022), 'ByteTrack: Multi-Object Tracking by Associating Every Detection Box,' ECCV — the low-score two-pass association trick; strong evidence that association strategy beats fancier embeddings.
- Ristani, Solera, Zou, Cucchiara & Tomasi (2016), 'Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking,' ECCV — the definition and motivation of IDF1 via global identity matching.
- Cao, Simon, Wei & Sheikh (2017), 'Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields (OpenPose),' CVPR — the bottom-up grouping via PAF line integrals; the classic constant-cost multi-person method.
- Zhang, Wang, Wang, Zeng & Liu (2021), 'FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking,' IJCV — why joint detection + embedding heads compete and how anchor-free, high-resolution design fixes it.
- Meinhardt, Kirillov, Leal-Taixe & Feichtenhofer (2022), 'TrackFormer: Multi-Object Tracking with Transformers,' CVPR — track queries and tracking-by-attention; the transformer-tracker paradigm.
- Nibali, He, Morgan & Prendergast (2018), 'Numerical Coordinate Regression with Convolutional Neural Networks (DSNT)' — differentiable soft-argmax decoding of heatmaps to sub-pixel coordinates.
