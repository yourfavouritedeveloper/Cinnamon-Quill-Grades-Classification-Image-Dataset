#  CinnamonNet

> **Benchmarking CNN Architectures and Training Strategies for Fine-Grained Spice Quality Classification**

Automated quality grading of agricultural commodities remains an open problem in precision agriculture, where fine-grained visual differences between grades demand more than standard image classification. We present a systematic empirical study on the classification of Ceylon cinnamon quills into four commercial grades — **Alba**, **C4**, **C5**, and **C5 Special** — using deep convolutional neural networks.

We benchmark two architectures (ResNet18, VGG16), two optimizers (SGD, Adam), and two initialization strategies (ImageNet pretraining vs. random initialization) across **eight controlled experiments**.

> **Key finding:** Overall accuracy improves by 21 percentage points under transfer learning (70.5% → 91.4%), but the benefit is not uniform — C4 grade improves by **53 percentage points** (0.36 → 0.89) while Alba remains largely unaffected (±1%). C4 and C5 are not just similar to each other — they are dissimilar in ways that ImageNet features can detect but task-specific features cannot learn from scratch without sufficient data.

---

## Table of Contents

- [Dataset](#dataset)
- [Method](#method)
- [Experiments](#experiments)
- [Results](#results)
- [Analysis](#analysis-and-discussion)
- [Reproduction](#reproduction)

---

## Dataset

### Grade Overview

The dataset consists of photographs of Ceylon cinnamon quills across four commercial grades defined by the Sri Lanka Standards Institution (SLSI):

| Grade | Description | Visual Characteristics |
|---|---|---|
| **Alba** | Highest quality | Tight, uniform roll; pale color; smooth surface |
| **C5 Special** | Premium sub-grade | Slightly looser; consistent diameter |
| **C5** | Standard grade | Visible texture variation; moderate diameter |
| **C4** | Lower grade | Irregular roll; surface imperfections; darker |

### Data Splits

| Split | Proportion | Purpose |
|---|---|---|
| Training | 70% | Model fitting |
| Validation | 15% | Hyperparameter selection, early stopping |
| Test | 15% | Final unbiased evaluation |

Splits are stratified by class and fixed with `seed=42`.

### Class Imbalance

The dataset is naturally imbalanced, reflecting real-world production distributions where Alba is rarer than C5. Training uses `WeightedRandomSampler` to assign each sample a weight inversely proportional to its class frequency.

### Preprocessing

All images are resized to **224 × 224** and normalized with ImageNet channel statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Training augmentation includes random horizontal/vertical flips, rotation (±15°), and random resized crop (scale 0.8–1.0). Validation and test sets receive only resize and normalization.

### Directory Structure

```
datasets/cinnamon/
├── Alba/
├── C4/
├── C5/
└── C5 Special/
```

---

## Method

### Architectures

#### ResNet18

ResNet18 introduces residual connections that bypass one or more layers:

```
Output = F(x, {Wᵢ}) + x
```

With **11.7M parameters**, it is compact enough to train from scratch on a small dataset. The final layer is replaced:

```
fc: Linear(512, 1000)  →  Linear(512, 4)
```

#### VGG16

VGG16 is a sequential architecture of 16 weight layers with no skip connections. Its **138M parameters** provide high representational capacity, but gradients degrade through depth, making it substantially harder to train from scratch. The classifier is modified:

```python
classifier[5] = Dropout(p=0.5)    # regularization
classifier[6] = Linear(4096, 4)   # output layer
```

#### Architecture Comparison

| Property | ResNet18 | VGG16 |
|---|---|---|
| Parameters | ~11.7M | ~138M |
| Skip connections | Yes | No |
| Overfitting risk (small data) | Low | High |
| Training from scratch | Feasible | Difficult |
| Transfer learning benefit | Moderate | High |

### Training Strategies

| Strategy | Description |
|---|---|
| `pretrained=True` | Backbone initialized with ImageNet weights; all layers fine-tuned at low LR |
| `pretrained=False` | All weights randomly initialized (Kaiming); learns from cinnamon data only |

### Optimizers

**SGD with Momentum**
```python
optimizer = SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(step_size=7, gamma=0.1)
```

**Adam**
```python
optimizer = Adam(lr=1e-4, weight_decay=1e-4)
scheduler = StepLR(step_size=7, gamma=0.1)
```

---

## Experiments

Eight experiments across the full factorial design:

| Experiment | Architecture | Optimizer | Pretrained |
|---|---|---|---|
| E1 | ResNet18 | Adam | ✅ Yes |
| E2 | ResNet18 | Adam | ❌ No |
| E3 | ResNet18 | SGD | ✅ Yes |
| E4 | ResNet18 | SGD | ❌ No |
| E5 | VGG16 | Adam | ✅ Yes |
| E6 | VGG16 | Adam | ❌ No |
| E7 | VGG16 | SGD | ✅ Yes |
| E8 | VGG16 | SGD | ❌ No |

> All experiments share: `batch_size=8`, `epochs=15`, `seed=42`, identical data splits.

---

## Results

### Overall Accuracy — ResNet18 + Adam

| Configuration | Accuracy | Weighted F1 |
|---|---|---|
| ResNet18 + Adam + **Pretrained** | **0.9137** | **0.9137** |
| ResNet18 + Adam + Scratch | 0.7050 | 0.7050 |
| Δ (transfer learning gain) | **+0.2087** | **+0.2087** |

### Class-wise Accuracy Breakdown

| Class | Pretrained | From Scratch | Δ |
|---|---|---|---|
| Alba | 0.9130 | 0.9231 | −0.010 |
| **C4** | **0.8929** | 0.3636 | **+0.529** |
| C5 | 0.9444 | 0.6667 | +0.278 |
| C5 Special | 0.9038 | 0.7463 | +0.158 |

### Full Experiment Summary

| Model | Pretrained | Optimizer | Accuracy | F1 | Alba | C4 | C5 | C5 Special |
|---|---|---|---|---|---|---|---|---|
| ResNet18 | ✅ | Adam | 0.9137 | 0.9137 | 0.913 | 0.893 | 0.944 | 0.904 |
| ResNet18 | ❌ | Adam | 0.7050 | 0.7050 | 0.923 | 0.364 | 0.667 | 0.746 |
| ResNet18 | ✅ | SGD | — | — | — | — | — | — |
| ResNet18 | ❌ | SGD | — | — | — | — | — | — |
| VGG16 | ✅ | Adam | — | — | — | — | — | — |
| VGG16 | ❌ | Adam | — | — | — | — | — | — |
| VGG16 | ✅ | SGD | — | — | — | — | — | — |
| VGG16 | ❌ | SGD | — | — | — | — | — | — |

*Run the full training suite to populate remaining rows.*

---

## Analysis and Discussion

### The Transfer Learning Gap Is Not Uniform

The headline result — **+21pp accuracy from pretraining** — conceals a more important pattern. Alba accuracy is virtually unchanged (±1%) while C4 improves by **53 percentage points**. This is not an artifact of class imbalance; `WeightedRandomSampler` equalizes class frequency during training.

The asymmetry implies something structural: **C4 images contain discriminative features that ImageNet filters can detect, but that a from-scratch model cannot reliably learn from the available samples.**

Alba, by contrast, is visually distinctive enough (tight, pale, uniform) that even randomly initialized features achieve high accuracy (92%). It is "easy" from the feature-learning perspective.

> In fine-grained agricultural classification, **the hardest class is not the rarest — it is the one whose discriminative features are the most subtle and the most dependent on prior visual knowledge.**

### Why C4 Is the Hard Class

C4 quills are visually intermediate. Their defining characteristics — slight surface irregularity, marginally looser roll geometry, subtle color shift — are encoded in **mid-frequency texture patterns**. These are precisely the patterns that layers 2–4 of a ResNet backbone (trained on ImageNet's diverse texture vocabulary) can detect, but that a from-scratch model needs far more data to learn.

### Limitations

- **Dataset size** — test set variance is significant; a single confusing sample can shift accuracy by several points
- **Single seed** — variance across random seeds is not reported; research-level claims require 3–5 seeds
- **No statistical testing** — differences are reported as point estimates without confidence intervals
- **Frozen vs. fully fine-tuned** — a systematic comparison of freezing strategies was not conducted



---

## Reproduction

### Installation

```bash
pip install torch torchvision tensorboard scikit-learn pillow
# Python 3.9+, PyTorch 2.x
```

### Run All 8 Experiments

```bash
python train.py
```

Trains all configurations, saves checkpoints to `checkpoints/`, logs to `runs/`.

### Monitor Training

```bash
tensorboard --logdir=runs
```

### Predict a Single Image

```bash
python predict.py \
    --image "datasets/cinnamon/C5/C5_01.JPG" \
    --model resnet18 \
    --checkpoint checkpoints/resnet18_adam_pretrained.pth
```

### Checkpoint Naming

```
checkpoints/
├── resnet18_adam_pretrained.pth
├── resnet18_adam_scratch.pth
├── resnet18_sgd_pretrained.pth
├── resnet18_sgd_scratch.pth
├── vgg16_adam_pretrained.pth
├── vgg16_adam_scratch.pth
├── vgg16_sgd_pretrained.pth
└── vgg16_sgd_scratch.pth
```

---

