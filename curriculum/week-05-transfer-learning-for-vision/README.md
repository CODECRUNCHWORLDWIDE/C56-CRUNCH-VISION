# Week 5 — Transfer learning for vision

> **Goal:** by Sunday you can (1) explain, in representation-learning and statistical terms, why features learned on one distribution transfer to another and where that guarantee degrades; (2) place any real problem on the feature-extraction-to-full-fine-tuning spectrum from a principled reading of data size, domain distance, and label budget; (3) fine-tune a modern backbone responsibly — matched preprocessing, discriminative and warmed-up learning rates, layer-wise decay, early stopping, and leakage-proof evaluation; (4) bound target-domain error with the H-divergence theory of domain adaptation (Ben-David et al., 2010); and (5) reason about the self-supervised and vision-language pretraining (CLIP, DINO, MAE) that now supplies the backbones you transfer from, including when linear probing beats fine-tuning and how to avoid distorting robust features.

Almost nobody trains a vision model from scratch anymore. They start from a network already trained on tens of millions — increasingly billions — of images and **transfer** its learned representation to their own task. This is the single highest-leverage technique in applied vision: it turns a ten-million-image problem into a five-hundred-image one, and it is how essentially every production image model is built. This week is the graduate treatment of that fact.

The undergraduate version says 'freeze the backbone, swap the head, maybe fine-tune.' We will go much further. You will see *why* transfer works as a statement about the geometry of learned representations — early filters converge to Gabor-like edge and colour detectors regardless of task (Zeiler & Fergus, 2014), and the transferability of a layer is measurable (Yosinski et al., 2014). You will see *when* it breaks, formalized by the domain-adaptation bound of Ben-David et al. (2010): target error is controlled by source error plus a divergence between the two feature distributions plus an adaptability term, and every practical trick either shrinks source error or shrinks that divergence. You will fine-tune the way strong labs actually do it — head-first, discriminative learning rates, layer-wise LR decay, cosine schedules with warmup — and you will evaluate with the paranoia that pretraining demands, because a backbone that already saw your test images inflates every number you report.

Finally you will meet the backbones themselves as objects of study. The ImageNet-supervised ResNet is no longer the only, or best, thing to transfer from: self-supervised models (SimCLR, MoCo, DINO), masked autoencoders (MAE), and vision-language models (CLIP) now produce representations that transfer further and more robustly — but they change the rules, sometimes making a linear probe outperform full fine-tuning (Kumar et al., 2022). By the end you will not just call `model.fc = nn.Linear(...)`; you will predict what transfer will buy you, and defend the strategy you chose.

## Learning objectives

By the end of this week, you will be able to:

- **Explain** transferability as a property of the learned representation — hierarchical, general-to-specific — and cite the empirical evidence (Yosinski 2014; Zeiler & Fergus 2014) for where a layer stops being reusable.
- **Adapt** a pretrained backbone to a new label set correctly: identify backbone vs. head, swap the head, and match the backbone's exact preprocessing and normalization statistics.
- **Choose** a point on the feature-extraction-to-fine-tuning spectrum from data size, domain distance, and label budget, and justify it against the four-quadrant heuristic and its failure cases.
- **Fine-tune** responsibly with discriminative and layer-wise-decayed learning rates, warmup, augmentation, weight decay, and early stopping — and diagnose catastrophic forgetting from the curves.
- **Bound** target-domain error using the H-divergence theory (Ben-David et al., 2010) and connect each term to a concrete engineering lever.
- **Evaluate** transfer honestly under small-data and pretraining-overlap risk, using held-out or cross-validated estimates and per-class error analysis.
- **Compare** supervised, self-supervised (DINO/MAE/SimCLR), and vision-language (CLIP) backbones as transfer sources, and predict when linear probing beats full fine-tuning.
- **Quantify** the data-efficiency of transfer by measuring accuracy versus images-per-class against a from-scratch baseline, and state the minimum viable dataset for a target task.

## Standards this week meets

| Bar | What this week is measured against |
| --- | --- |
| University | `CS 231N` — reuse a pretrained representation on a new task by feature extraction or fine-tuning, and justify the choice from data size and domain distance. |
| Industry | Decide whether to freeze a pretrained backbone or fine-tune it on the data a project actually has, and defend the decision in writing to the team that has to live with it. |
| Beyond the bar | bounds target-domain error with the H-divergence result and then has the learner measure that divergence on their own two datasets instead of citing it — `exercises/exercise-04-measure-domain-divergence.md` |

## Prerequisites

- Week 3-4: CNN architecture, the training loop, augmentation, regularization, and honest evaluation (train/val/test, confusion matrices, per-class metrics).
- Comfort loading models from `torchvision.models` (or `timm`) and reading a model's module tree.
- The previous week's mini-project (an end-to-end image classifier), committed and working.

## This week

**Lectures**

1. [Lecture 1 — Why transfer works: the geometry of learned representations](lecture-notes/01-why-transfer-works.md)
2. [Lecture 2 — Feature extraction vs. fine-tuning: a decision theory](lecture-notes/02-feature-extraction-vs-finetuning.md)
3. [Lecture 3 — Fine-tuning responsibly: preprocessing, schedules, and honest evaluation](lecture-notes/03-finetuning-in-practice.md)
4. [Lecture 4 — A theory of transferability: the domain-adaptation bound](lecture-notes/04-theory-of-transferability-domain-adaptation.md)
5. [Lecture 5 — What you transfer from now: self-supervised and vision-language backbones](lecture-notes/05-self-supervised-and-foundation-backbones.md)

**Exercises**

1. [Exercise 1 — Load a backbone, swap the head, and get the preprocessing right](exercises/exercise-01-load-and-swap-head.md)
2. [Exercise 2 — Train a frozen feature extractor (linear probe)](exercises/exercise-02-feature-extraction.md)
3. [Exercise 3 — Fine-tune with discipline and compare strategies](exercises/exercise-03-finetune-and-compare.md)
4. [Exercise 4 — Measure the domain divergence and read the adaptation bound](exercises/exercise-04-measure-domain-divergence.md)

**Challenges**

1. [Challenge 1 — Push transfer to a different domain, and explain the gap with theory](challenges/challenge-01-domain-shift.md)
2. [Challenge 2 — How little data can you win with?](challenges/challenge-02-tiny-data.md)
3. [Challenge 3 — Fine-tune without wrecking robustness (LP-FT and WiSE-FT)](challenges/challenge-03-robust-finetuning.md)

**Then:** the [mini-project](mini-project/) is the week's real build. Finish it before Week 6.

**Graduate track:** [problem set](problem-set.md) · [reading list](reading-list.md)

## Deliverable

A pretrained backbone adapted to a small custom dataset two ways — as a frozen feature extractor and with disciplined fine-tuning (discriminative/layer-wise LRs, warmup, augmentation, early stopping) — with a rigorous comparison of accuracy, training time, and the train/val gap, per-class metrics and a confusion matrix for the chosen model, a pretraining-overlap and leakage audit, and a written recommendation of which to ship that cites the data-size x domain-distance reasoning and, where relevant, the H-divergence framing.

See also: [homework](homework.md) · [quiz](quiz.md) · [resources](resources.md).
