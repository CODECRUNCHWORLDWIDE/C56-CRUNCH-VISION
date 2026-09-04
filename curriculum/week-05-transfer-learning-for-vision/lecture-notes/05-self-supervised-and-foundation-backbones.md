# Lecture 5 — What you transfer from now: self-supervised and vision-language backbones

The ImageNet-supervised ResNet was the default transfer source for a decade, but it is no longer the
best — or even the most common — thing to transfer from. Modern backbones are pretrained *without task labels*,
on far more data, with objectives designed to produce representations that transfer further and more robustly.
Knowing how they are built tells you when to prefer them and how the transfer rules change. This is the
state-of-the-art layer on top of the classical picture.

## Why move past supervised ImageNet pretraining

Supervised pretraining ties the representation to 1000 fixed categories: the model learns exactly the features
that separate *those* classes, which can discard information useful elsewhere. It also needs millions of
*labeled* images — the expensive resource. Self-supervised learning (SSL) removes the label bottleneck by
inventing a *pretext* task solvable from the image alone, and empirically yields features that are more general
and often more robust to distribution shift.

## Contrastive SSL: SimCLR, MoCo, DINO

**Contrastive** methods learn an embedding in which two augmentations of the *same* image are pulled together
and different images pushed apart. SimCLR (Chen et al., 2020, ICML) formalized this with the **InfoNCE** loss

    L = - log [ exp(sim(z_i, z_j)/tau) / sum_{k != i} exp(sim(z_i, z_k)/tau) ],

where `z_i, z_j` are projections of two views of one image, `sim` is cosine similarity, and `tau` is a
temperature. It needs large batches (many negatives); MoCo (He et al., 2020, CVPR) removed that constraint with
a momentum-updated key encoder and a queue of negatives. **DINO** (Caron et al., 2021, "Emerging Properties in
Self-Supervised Vision Transformers", ICCV) used self-distillation with no negatives — a student network
matches a momentum teacher's output on different views — and, strikingly, its attention maps segment objects
without ever being told what an object is. DINO/DINOv2 features are among the strongest available for *frozen*
transfer: a **linear probe** on them is a formidable baseline, which is exactly the regime (Lecture 2) where
freezing beats fine-tuning.

## Masked image modeling: MAE and BEiT

The other family is **generative / masked**. The **Masked Autoencoder** (He et al., 2022, "Masked Autoencoders
Are Scalable Vision Learners", CVPR) masks ~75% of image patches and trains a ViT to reconstruct the missing
pixels from the visible ones. Because most of the image is hidden, the model must learn semantic structure to
inpaint, and because the encoder only sees the 25% visible patches, pretraining is cheap. MAE features tend to
*need fine-tuning* to shine (weaker linear-probe, strong fine-tuned) — the opposite tendency to DINO — which is
itself a useful signal: masked-generative pretraining and contrastive pretraining sit differently on the
Lecture-2 spectrum. This is a concrete case where "which strategy?" depends on *which backbone*, not just your
data.

## Vision-language pretraining: CLIP

**CLIP** (Radford et al., 2021, "Learning Transferable Visual Models From Natural Language Supervision", ICML)
trains an image encoder and a text encoder jointly on ~400M web (image, caption) pairs with a contrastive loss
that aligns an image with its caption against other captions in the batch. The payoff is a representation
grounded in language, which enables **zero-shot classification**: to classify into new categories, embed the
class names as text prompts and pick the nearest to the image embedding — *no* labeled target images at all.
CLIP features are exceptionally strong under distribution shift, and CLIP is now a default transfer source. But
it sharpens the Lecture-2/Kumar et al. warning: naive full fine-tuning of CLIP can *distort* its robust
features and hurt out-of-distribution accuracy.

## Robust fine-tuning: don't wreck what you started with

Because these backbones start with genuinely good, robust features, how you fine-tune matters more than ever.
Two techniques you should know:

- **LP-FT** (Kumar et al., 2022): linear-probe to a good head *first*, then fine-tune end-to-end. Fine-tuning
  from a random head drags the features toward fitting noise before the head is any good; probing first gives
  the fine-tune a sane starting point and preserves out-of-distribution accuracy.
- **WiSE-FT** (Wortsman et al., 2022, "Robust fine-tuning of zero-shot models", CVPR): **interpolate the weights**
  of the zero-shot (pre-fine-tune) model and the fine-tuned model, `theta = (1-alpha) theta_zeroshot + alpha
  theta_finetuned`. This simple weight averaging recovers much of fine-tuning's in-distribution gain while
  keeping the zero-shot model's out-of-distribution robustness — a nearly free lunch, and a vivid demonstration
  that the pretrained weights are a resource to be *preserved*, not just a starting point to be overwritten.

## How this rewires the decision from Lecture 2

- **DINO/DINOv2, CLIP:** excellent frozen features -> *prefer linear probing / feature extraction*, especially
  on small data or when out-of-distribution robustness matters. Fine-tune only with care (LP-FT, WiSE-FT).
- **MAE:** weaker frozen, strong fine-tuned -> *fine-tune* if you have the data.
- **Data efficiency:** SSL/vision-language backbones push the minimum-viable-dataset (Challenge 2) far lower
  than supervised ImageNet backbones did, because their representations are more general.

## Pitfalls

- **Using ImageNet normalization on a CLIP/SSL backbone.** Wrong statistics, silent failure (Lecture 3).
- **Full-fine-tuning a robust backbone naively.** Distorts features; prefer LP-FT or WiSE-FT for
  shift-robustness.
- **Assuming linear-probe vs. fine-tune has one answer.** It depends on the pretraining objective: contrastive/
  language backbones favour probing, masked-generative favour fine-tuning.

**Takeaway:** the backbones you transfer from are now largely self-supervised (SimCLR/MoCo/DINO contrastive; MAE
masked) or vision-language (CLIP), pretrained without task labels on far more data, yielding more general and
more robust features. Which transfer strategy wins depends on the pretraining objective — contrastive/language
models excel at frozen linear probing, masked-generative models at fine-tuning — and robust fine-tuning (LP-FT,
WiSE-FT) exists specifically to keep from distorting features that were good to begin with.
