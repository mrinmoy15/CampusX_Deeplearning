# Transfer Learning

> **Transfer learning** is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem.

---

## Why Transfer Learning?

Training deep learning models from scratch requires:
- Massive amounts of labeled data
- Huge computational resources
- Long training time

Transfer learning solves this by **reusing a pre-trained model** as a starting point.

---

## Drivers of ML Success in Industry

The graph below shows how different ML paradigms contribute to commercial success over time:

| Paradigm | Commercial Trajectory |
|---|---|
| **Supervised Learning** | Early and steep rise — dominant for a long time |
| **Transfer Learning** | Late but highest potential — steepest long-term growth |
| **Unsupervised Learning** | Gradual rise, moderate commercial impact |
| **Reinforcement Learning** | Slowest to commercialize, still emerging |

**Key Insight:** Transfer learning is the *future driver* of ML success in industry — it surpasses supervised learning in commercial impact over time.

---

## How Transfer Learning Works

```
Pre-trained Model                  Target Task
(e.g. ImageNet, BERT)              (e.g. Medical Imaging, Sentiment)

  ┌─────────────────┐               ┌──────────────────┐
  │  Learned         │    Reuse /    │  Fine-tune on    │
  │  Features /      │ ──────────→  │  new smaller     │
  │  Weights         │   Transfer   │  dataset         │
  └─────────────────┘               └──────────────────┘
```

---

## Types of Transfer Learning

### 1. Feature Extraction
- Freeze all layers of the pre-trained model
- Only train the final classification head
- Use when: your dataset is **small** and **similar** to the source domain

### 2. Fine-Tuning
- Unfreeze some (or all) layers of the pre-trained model
- Retrain with a **very small learning rate**
- Use when: your dataset is **larger** or **different** from the source domain

### 3. Domain Adaptation
- Source and target domains are **different** but task is the same
- Model adapts its representation to the new domain

---

## When to Use Transfer Learning?

| Scenario | Recommendation |
|---|---|
| Small dataset, similar domain | Feature extraction (freeze layers) |
| Small dataset, different domain | Risky — try fine-tuning last few layers |
| Large dataset, similar domain | Fine-tune the full model |
| Large dataset, different domain | Train from scratch or full fine-tuning |

---

## Popular Pre-trained Models

| Domain | Model |
|---|---|
| **Computer Vision** | VGG, ResNet, InceptionNet, EfficientNet, ViT |
| **NLP** | BERT, GPT, RoBERTa, T5 |
| **Audio** | Wav2Vec, Whisper |
| **Multimodal** | CLIP, DALL-E |

---

## Advantages

- Requires **less labeled data**
- **Faster training** — converges quicker
- **Better performance** — especially on small datasets
- Reduces **computational cost**

## Limitations

- Pre-trained model may have **biases** from source data
- **Negative transfer** — if source and target domains are too different, performance can degrade
- Large models can be **memory intensive**

---

## Key Terms

| Term | Meaning |
|---|---|
| **Source Domain** | The original task the model was trained on |
| **Target Domain** | The new task you want to solve |
| **Frozen Layers** | Layers whose weights are not updated during training |
| **Fine-tuning** | Retraining some layers on the new task |
| **Negative Transfer** | When transferred knowledge hurts performance |
