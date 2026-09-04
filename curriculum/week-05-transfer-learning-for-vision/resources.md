# Week 5 — Resources

Curated, free where possible. You do not need all of these — pick what fits how you learn.

- [Stanford CS231n — Transfer Learning](https://cs231n.github.io/transfer-learning/) — the four quadrants and the head-first recipe, clearly explained.
- [PyTorch — Transfer Learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — feature extraction vs. fine-tuning in code, with correct preprocessing.
- [torchvision.models documentation](https://pytorch.org/vision/stable/models.html) — the model zoo, weight enums, and their bundled `transforms()`; the authoritative source for matched preprocessing.
- [`timm` (PyTorch Image Models) documentation](https://huggingface.co/docs/timm) — the broadest set of modern backbones (ConvNeXt, ViT, DINOv2, CLIP-init) with consistent fine-tuning and feature-extraction APIs.
- [OpenAI CLIP repository and model card](https://github.com/openai/CLIP) — zero-shot classification, the correct CLIP preprocessing statistics, and prompt templates.
- fast.ai course — practical fine-tuning, discriminative learning rates, and the head-first recipe, hands-on.
- [Hugging Face PEFT library](https://huggingface.co/docs/peft) — adapters, LoRA, and BitFit for parameter-efficient fine-tuning; the modern middle path for the small-data-different-domain corner.
- Ben-David et al. (2010), 'A theory of learning from different domains' — the primary source for the domain-adaptation bound; read Sections 2-3 alongside Lecture 4.

> All external links are references, not endorsements. Prefer primary sources (papers, official docs) when they exist.
