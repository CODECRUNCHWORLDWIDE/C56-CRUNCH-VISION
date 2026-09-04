# C56 · Crunch Vision

> A free, open-source 12-week course that builds computer vision from the pixel up — filtering, features, CNNs, detection, segmentation, tracking, video, and Vision Transformers — every idea shown on real images before you trust a metric.

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](LICENSE)
[![Python · PyTorch · OpenCV](https://img.shields.io/badge/stack-Python_·_PyTorch_·_OpenCV-14B8A6.svg)](#stack)
[![Built in the open](https://img.shields.io/badge/built-in%20the%20open-14B8A6.svg)](https://github.com/CODECRUNCHWORLDWIDE)

C56 is a Tier-2 computer-vision course, sized full (12 weeks) because it walks the whole road — from "what actually is a pixel, and what is a color?" to building, training, and deploying classifiers, detectors, segmenters, trackers, and Vision Transformers on real images and video. You will implement **convolution, edge detection, and non-max suppression by hand** before you lean on the library, so no layer is ever a black box. Everything is in **Python + PyTorch + OpenCV**, runs on a laptop CPU (a GPU only makes it faster), and pairs naturally with [C53 Crunch Nets](../C53-CRUNCH-NETS/) for the deep-learning foundations, [C5 Crunch AI & Data Science](../C5-CRUNCH-AI-DATA-SCIENCE/) for the data work, and [C28 Crunch MLOps](../C28-CRUNCH-MLOPS/) for shipping what you build.

---

## Standards & equivalency

> C56 stands in for a university's computer vision course.

**University equivalent.** Computer Vision — `CAP 4410`, `CS 4670`, `CS 231N`. Coverage: full. Every outcome of that course is mapped below to a week that assigns work on it, so no part of the claim rests on a topic the course only mentions.

C56 carries no credit, no transcript entry, no accreditation and no proctored exam. The equivalence is one of **content and skill**: image formation through Vision Transformers, taught here at the same depth or deeper, and assessed. What a registrar records is not something an open repository can give you.

| University outcome | Where this course teaches it | Depth |
| --- | --- | --- |
| Explain image formation, and treat a digital image as the sampling and quantization of a continuous signal, with its color representation | [Week 01](curriculum/week-01-images-pixels-color-and-filtering/) | deeper |
| Apply linear filtering and convolution to images, and reason about them in the frequency domain — sampling, aliasing, and image pyramids | [Week 01](curriculum/week-01-images-pixels-color-and-filtering/) | deeper |
| Detect edges, corners and scale-invariant interest points, and describe them with local feature descriptors | [Week 02](curriculum/week-02-classical-cv-edges-features-descriptors/) | deeper |
| Match features between two views and estimate the geometric transformation between them robustly, in the presence of outliers | [Week 02](curriculum/week-02-classical-cv-edges-features-descriptors/) | same |
| Explain the architecture, forward pass and backward pass of a convolutional neural network, and train one | [Week 03](curriculum/week-03-convolutional-neural-networks-for-vision/) | deeper |
| Build an image classifier end to end — data pipeline, loss, optimization, regularization — and evaluate it with the appropriate metrics | [Week 04](curriculum/week-04-image-classification-end-to-end/) | deeper |
| Reuse a pretrained representation on a new task by feature extraction or fine-tuning, and justify the choice | [Week 05](curriculum/week-05-transfer-learning-for-vision/) | deeper |
| Localize objects: box representation, IoU, non-maximum suppression, detector architectures, and mean Average Precision | [Week 06](curriculum/week-06-object-detection/) | deeper |
| Produce dense per-pixel predictions — semantic and instance segmentation — and measure them with region-overlap metrics | [Week 07](curriculum/week-07-semantic-and-instance-segmentation/) | deeper |
| Estimate keypoints and human pose, and track multiple objects across frames | [Week 08](curriculum/week-08-keypoints-pose-and-object-tracking/) | deeper |
| Estimate motion between frames with optical flow, and recognize activity in video | [Week 09](curriculum/week-09-video-understanding-and-optical-flow/) | same |
| Explain and apply attention-based vision models, and compare them against convolutional ones on evidence | [Week 10](curriculum/week-10-vision-transformers/) | deeper |
| Account for the computational cost of a vision model and adapt it to a stated deployment constraint | [Week 11](curriculum/week-11-deploying-vision-on-the-edge/) | deeper |
| Carry out a substantial vision project on a problem of the learner's own choosing, evaluate it, and present and defend it | [Week 12](curriculum/week-12-capstone-a-computer-vision-system/) | deeper |

Every row above points at a week that **assigns work** on that outcome — four exercises, three challenges, a mini-project, homework, a graduate problem set and a quiz — not a week that merely mentions it.

**The industry bar.** What an employer expects of somebody paid to build vision systems, and — plainly — where this course does it and where it does not.

| What the job expects | Where this course does it |
| --- | --- |
| Work lands as a commit in a repository you own | **Not taught.** No week names a version-control step. The nearest thing is the capstone's requirement that the result be a repository somebody else can clone and reproduce from its README — [`curriculum/week-12-capstone-a-computer-vision-system/mini-project/README.md`](curriculum/week-12-capstone-a-computer-vision-system/mini-project/README.md) |
| You read code you did not write and form a judgement on it | **Not as a code review.** No unit puts another person's implementation in front of the learner for a verdict. What the course does instead is hold every hand-written implementation to a reference one — [`exercise-03-convolution-by-hand.md`](curriculum/week-01-images-pixels-color-and-filtering/exercises/exercise-03-convolution-by-hand.md), [`exercise-01-iou-and-nms.md`](curriculum/week-06-object-detection/exercises/exercise-01-iou-and-nms.md) — and put the primary papers, not secondary summaries, in front of the learner every week ([`reading-list.md`](curriculum/week-06-object-detection/reading-list.md)) |
| Tests exist, and the command to run them is written down | **No test suite ships and no test runner is named.** Correctness is checked by parity assertions written into the builds instead: a numerical gradient check on the from-scratch convolution backward pass ([`exercise-04-conv-backward-from-scratch.md`](curriculum/week-03-convolutional-neural-networks-for-vision/exercises/exercise-04-conv-backward-from-scratch.md)), patch embedding implemented twice and asserted equal ([`exercise-01-patch-embedding.md`](curriculum/week-10-vision-transformers/exercises/exercise-01-patch-embedding.md)), and export output-and-preprocessing parity against the source model ([`exercise-03-export-and-verify.md`](curriculum/week-11-deploying-vision-on-the-edge/exercises/exercise-03-export-and-verify.md)) |
| You read failure instead of guessing | `## Common pitfalls` on twenty-five lecture pages names the failure and its cause, and Week 04 has the learner produce five runs each broken in one named way and build the symptom-to-cause-to-fix table from the curves ([`challenge-02-diagnose-a-broken-run.md`](curriculum/week-04-image-classification-end-to-end/challenges/challenge-02-diagnose-a-broken-run.md)). Those pitfalls are written as prose; the course does not quote captured exception text |
| Dependencies are isolated per project, and a formatter, a linter and a pipeline run over the work | **Not taught anywhere in this course.** The stack is named in this README and nothing in the twelve weeks installs, isolates, formats, lints or builds it |
| Cost is measured, not guessed | The roofline analysis, median and p95/p99 latency, peak memory and on-disk size in [`curriculum/week-11-deploying-vision-on-the-edge/mini-project/`](curriculum/week-11-deploying-vision-on-the-edge/mini-project/), plus the timed inference and accuracy-per-compute trade in [`challenge-02-speed-accuracy.md`](curriculum/week-06-object-detection/challenges/challenge-02-speed-accuracy.md) |
| It runs from a clean clone by following the README | [`curriculum/week-12-capstone-a-computer-vision-system/mini-project/`](curriculum/week-12-capstone-a-computer-vision-system/mini-project/) |
| A system's limits are documented for the people it affects | The model card, datasheet and drift plan in [`exercise-04-model-card-datasheet-and-drift-plan.md`](curriculum/week-12-capstone-a-computer-vision-system/exercises/exercise-04-model-card-datasheet-and-drift-plan.md) |
| The professional task is named, not implied | The `## Standards this week meets` block in every week's `README.md` |

**Beyond both bars.** Clearing the two floors is entry, not success. Open any of these and check in under a minute.

| What we add | Which bar it beats | Where it lives |
| --- | --- | --- |
| Every week's quiz publishes its full key in the same file, each answer carrying the reasoning and not only the letter, so nothing is withheld until a deadline | both | [`curriculum/week-01-images-pixels-color-and-filtering/quiz.md`](curriculum/week-01-images-pixels-color-and-filtering/quiz.md) |
| A graduate problem set every week, with worked solution sketches published beside the questions in eleven of the twelve weeks | university | [`curriculum/week-01-images-pixels-color-and-filtering/problem-set.md`](curriculum/week-01-images-pixels-color-and-filtering/problem-set.md) |
| Every core operation is built by hand before a library is allowed near it — convolution and its backward pass, Canny, IoU and NMS, patch embedding — and then cross-checked against the library result | both | [`curriculum/week-03-convolutional-neural-networks-for-vision/exercises/exercise-04-conv-backward-from-scratch.md`](curriculum/week-03-convolutional-neural-networks-for-vision/exercises/exercise-04-conv-backward-from-scratch.md) |
| Five lectures a week, where the fifth goes past the syllabus into current research — set prediction without anchors, RAFT, Mask2Former, self-supervised backbones, compilers and TinyML | university | [`curriculum/week-06-object-detection/lecture-notes/05-detr-set-prediction.md`](curriculum/week-06-object-detection/lecture-notes/05-detr-set-prediction.md) |
| Failure is manufactured on purpose: five training runs each broken in one named way, and a symptom-to-cause-to-fix guide built from the curves they produce | industry | [`curriculum/week-04-image-classification-end-to-end/challenges/challenge-02-diagnose-a-broken-run.md`](curriculum/week-04-image-classification-end-to-end/challenges/challenge-02-diagnose-a-broken-run.md) |
| Privacy, consent and refusal are assessed rather than mentioned — a lawful-basis statement in the tracking week, and a written legal and ethical defense in the capstone | both | [`curriculum/week-12-capstone-a-computer-vision-system/challenges/challenge-02-demo-and-defend.md`](curriculum/week-12-capstone-a-computer-vision-system/challenges/challenge-02-demo-and-defend.md) |

**Gaps we declare.** None against the outcome set above. Two kinds of gap sit outside it and are worth naming anyway. First, in subject: C56 does not teach three-dimensional vision — camera calibration, stereo, structure from motion, or multi-view geometry past a two-view homography — nor generative image models, and it does not claim them. Second, in professional practice: as the industry table says plainly, the course teaches no version control, no environment isolation, no formatter or linter, no pipeline, and ships no runnable test suite; a learner needs those from elsewhere before the capstone is production work rather than a portfolio piece.

---

## Pathway summary

- **Full-time:** 12 weeks · ~24 hrs/week · ~288 hours
- **Working-engineer pace:** 6 months · ~12 hrs/week
- **Evening pace:** 12 months · ~6 hrs/week

See [`SYLLABUS.md`](SYLLABUS.md).

---

## What you will be able to do at the end of 12 weeks

- **Reason about images as data:** explain pixels, color spaces, channels, and resolution, and load, transform, and visualize images fluently.
- **Filter and transform images:** implement convolution, blurring, sharpening, and gradients by hand, and know when to reach for the classical tool.
- **Extract classical features:** find edges, corners, and keypoints, and match images with descriptors — the pre-deep-learning toolkit that still ships in production.
- **Build CNNs for vision:** implement convolution and pooling, then design and train convolutional classifiers that generalize.
- **Use transfer learning:** fine-tune pretrained backbones on your own small datasets and know when to freeze vs. fine-tune.
- **Detect objects:** understand anchors, IoU, and non-max suppression, and train and evaluate a detector with mAP.
- **Segment images:** build semantic and instance segmentation and measure them with the right metrics.
- **Track and understand motion:** estimate keypoints and pose, track objects across frames, and reason about optical flow and video models.
- **Build Vision Transformers:** implement patch embedding and self-attention for images and compare ViTs to CNNs honestly.
- **Deploy vision on the edge:** export, quantize, and benchmark a model to run in real time on constrained hardware.
- **Do it honestly:** measure on held-out images, report failure modes and dataset bias, and respect image licenses, privacy, and consent.

---

## The 12 weeks

| # | Week | Focus |
|---|------|-------|
| 1 | [Week 1 — Images, pixels, color & filtering](curriculum/week-01-images-pixels-color-and-filtering/) | The image as a sampled, quantized signal; color as tristimulus; filtering as an LSI system — with the theory underneath. |
| 2 | [Week 2 — Classical CV: edges, features, descriptors](curriculum/week-02-classical-cv-edges-features-descriptors/) | Edges, corners, keypoints, descriptors, matching, and the robust geometry underneath — derived, not just called. |
| 3 | [Week 3 — Convolutional neural networks for vision](curriculum/week-03-convolutional-neural-networks-for-vision/) | Convolution as a learned equivariant operator — its algebra, its backward pass, and the architectures built from it. |
| 4 | [Week 4 — Image classification end to end](curriculum/week-04-image-classification-end-to-end/) | The full classification pipeline — data, metrics, regularization, training dynamics — with the theory underneath. |
| 5 | [Week 5 — Transfer learning for vision](curriculum/week-05-transfer-learning-for-vision/) | Reuse pretrained backbones — feature extraction, fine-tuning, and the theory of when transfer holds. |
| 6 | [Week 6 — Object detection](curriculum/week-06-object-detection/) | Boxes, IoU, NMS, detector architectures, and COCO mAP — derived, not hand-waved. |
| 7 | [Week 7 — Semantic & instance segmentation](curriculum/week-07-semantic-and-instance-segmentation/) | Dense prediction — semantic, instance, panoptic — with the architectures, losses, and metrics derived, not just named. |
| 8 | [Week 8 — Keypoints, pose & object tracking](curriculum/week-08-keypoints-pose-and-object-tracking/) | Pose as heatmap regression, and multi-object tracking as Bayesian filtering + optimal assignment — with the theory underneath. |
| 9 | [Week 9 — Video understanding & optical flow](curriculum/week-09-video-understanding-and-optical-flow/) | Motion fields from first principles, and the spacetime architectures that read time — with the math and the cost. |
| 10 | [Week 10 — Vision Transformers](curriculum/week-10-vision-transformers/) | Attention over patches: the architecture, its complexity, its data-hunger, and the SSL/multimodal frontier — with the math underneath. |
| 11 | [Week 11 — Deploying vision on the edge](curriculum/week-11-deploying-vision-on-the-edge/) | Make a trained vision model small, fast, portable, and honestly benchmarked on constrained hardware — with the theory underneath. |
| 12 | [Week 12 — Capstone: a computer-vision system](curriculum/week-12-capstone-a-computer-vision-system/) | Ship one honestly-evaluated, deployed vision system — with the statistics and ethics to defend it. |

---

## Stack

**Python 3.11+**, **PyTorch 2.x** + **torchvision**, **OpenCV**, **NumPy**, **Pillow**, **matplotlib**, and a notebook environment (Jupyter or VS Code). Datasets are small and public (MNIST, CIFAR-10, Oxford Pets, a handful of COCO images, short clips) so everything runs on a CPU in minutes; a GPU (local or a free cloud notebook) is optional and only speeds training up. No paid APIs, no proprietary tools.

## How each week is structured

Every week ships the same parts: **lecture notes** (the ideas), **exercises**
(guided, one concept each), **challenges** (open-ended, you make the calls), a
**mini-project** (the week's real build), a **quiz** (check yourself), and
**homework + resources**. Do them in that order.

## Ethics & honest practice

Vision is the most privacy-laden branch of AI — it looks at people, places, and documents. Four rules run through this course: **(1) Measure on images the model has never seen** — a number on training data is not a result. **(2) Report failure modes and dataset bias** — every model card names where the model breaks and for whom, and vision datasets are notoriously skewed by geography, skin tone, and lighting. **(3) Respect provenance, licenses, and consent** — you only train on images you are allowed to use, and faces and personal data deserve special care. **(4) Do not build surveillance you would not want aimed at you** — capability without conscience is not skill.

---

_Part of [Code Crunch Worldwide](https://github.com/CODECRUNCHWORLDWIDE) — free, open-source, built in the open._
