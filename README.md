# Cinnamon Quill Grade Classification

A PyTorch deep learning pipeline for classifying cinnamon quill images into four commercial quality grades: **Alba**, **C4**, **C5**, and **C5 Special**.

This project systematically compares two CNN architectures (ResNet18, VGG16) across two optimizers (SGD, Adam) and two training strategies (transfer learning, from scratch) — yielding **8 distinct experimental configurations**.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Architectures](#architectures)
6. [Training Configurations](#training-configurations)
7. [Running Experiments](#running-experiments)
8. [Results](#results)
9. [Prediction](#prediction)
10. [Analysis](#analysis)

---

## Overview

| Factor | Options |
|---|---|
| Architecture | ResNet18, VGG16 |
| Optimizer | SGD, Adam |
| Pretrained weights | Yes (ImageNet), No (from scratch) |
| Classes | Alba, C4, C5, C5 Special |
| Input size | 224 × 224 |
| Loss function | CrossEntropyLoss |

Each combination is a separate experiment. Results are logged to TensorBoard and a final summary table is printed after all runs complete.

---

## Project Structure

```
project/
├── datasets/
│   └── cinnamon/
│       ├── Alba/
│       ├── C4/
│       ├── C5/
│       └── C5 Special/
│
├── checkpoints/              # Saved .pth model files
├── runs/                     # TensorBoard logs
│
├── models/
│   └── cinnamon_model.py     # ResNet18 and VGG16 factory functions
│
├── train.py                  # Runs all 8 experiments, prints summary
├── predict.py                # Predict grade for a single image
├── utils.py                  # Transforms, dataset loading, metrics
└── README.md
```

---

## Installation

```bash
pip install torch torchvision tensorboard scikit-learn pillow
```

Tested with Python 3.9+, PyTorch 2.x.

---

## Dataset

Place images in class-named subdirectories:

```
datasets/cinnamon/
    Alba/         ← highest grade
    C4/
    C5/
    C5 Special/   ← specialty grade
```

The dataset is automatically split at runtime:

| Split | Proportion |
|---|---|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

**Class imbalance** is addressed via `WeightedRandomSampler`, which oversamples minority classes during training.

### Preprocessing

All images are resized to 224 × 224 and normalized with ImageNet statistics:

```
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Augmentation (training only)

- `RandomHorizontalFlip`
- `RandomVerticalFlip`
- `RandomRotation(15°)`
- `RandomResizedCrop(224, scale=(0.8, 1.0))`

---

## Architectures

Both architectures are defined in `models/cinnamon_model.py`.

### ResNet18

A residual network with 18 layers. The final fully-connected layer is replaced:

```
Original:  fc → Linear(512, 1000)
Modified:  fc → Linear(512, 4)
```

Residual skip connections make ResNet18 easier to optimize from scratch. Gradient flow is preserved across all layers, so even random-initialized training reaches reasonable accuracy.

### VGG16

A deep sequential network with 16 weight layers. The classifier head is replaced and regularized:

```
Original:  classifier[6] → Linear(4096, 1000)
Modified:  classifier[5] → Dropout(0.5)          ← added for regularization
           classifier[6] → Linear(4096, 4)
```

VGG16 has ~138M parameters vs ResNet18's ~11M. It benefits more from pretrained weights and is substantially slower to train from scratch.

### Transfer Learning vs. From Scratch

```python
# Transfer learning — loads ImageNet weights, fine-tunes on cinnamon data
model = get_resnet18(pretrained=True)

# From scratch — random initialization, trains entirely on cinnamon data
model = get_resnet18(pretrained=False)
```

When `pretrained=True`, the backbone retains learned feature detectors (edges, textures, shapes) from ImageNet. Only the final classifier is reinitialized. This is especially impactful when your dataset is small.

---

## Training Configurations

### Optimizer Comparison

| Property | SGD | Adam |
|---|---|---|
| Update rule | Gradient descent + momentum | Adaptive per-parameter learning rates |
| Hyperparameters | `lr=0.001`, `momentum=0.9` | `lr=0.0001` |
| Scheduler | `StepLR(step=7, γ=0.1)` | `StepLR(step=7, γ=0.1)` |
| Typical behavior | Stable convergence, may need tuning | Faster convergence, less sensitive to lr |

### Shared Settings

| Hyperparameter | Value |
|---|---|
| Batch size | 8 |
| Epochs | 15 |
| Loss function | CrossEntropyLoss |
| Sampler | WeightedRandomSampler |
| Dropout (VGG only) | 0.5 |

---

## Running Experiments

### Run all 8 experiments

```bash
python train.py
```

This trains every combination of architecture × optimizer × pretrained and prints a final comparison table. Each model is saved to `checkpoints/`.

### TensorBoard

```bash
tensorboard --logdir=runs
```

Each run is logged under a separate tag, e.g. `resnet18_adam_pretrained`, so loss/accuracy curves can be compared side by side.

### Checkpoint naming

```
checkpoints/
    resnet18_sgd_pretrained.pth
    resnet18_sgd_scratch.pth
    resnet18_adam_pretrained.pth
    resnet18_adam_scratch.pth
    vgg16_sgd_pretrained.pth
    vgg16_sgd_scratch.pth
    vgg16_adam_pretrained.pth
    vgg16_adam_scratch.pth
```

---

## Results

### Observed Results (ResNet18 + Adam)

| Pretrained | Accuracy | F1 Score |
|---|---|---|
| Yes (transfer learning) | **0.9137** | **0.9137** |
| No (from scratch) | 0.7050 | 0.7050 |

Transfer learning improved accuracy by **~21 percentage points** on ResNet18 + Adam.

### Class-wise Accuracy

| Class | Pretrained | From Scratch |
|---|---|---|
| Alba | 0.9130 | 0.9231 |
| C4 | **0.8929** | 0.3636 |
| C5 | **0.9444** | 0.6667 |
| C5 Special | **0.9038** | 0.7463 |

From scratch struggles most with **C4**, which is visually similar to C5. Pretrained features resolve this ambiguity because ImageNet already encodes fine texture discrimination.

### Full Experiment Summary Table

After running `train.py`, a table like this is printed to stdout:

| Model | Pretrained | Optimizer | Loss | Accuracy | F1 | Alba | C4 | C5 | C5 Special |
|---|---|---|---|---|---|---|---|---|---|
| ResNet18 | Yes | Adam | CE | 0.9137 | 0.9137 | 0.913 | 0.893 | 0.944 | 0.904 |
| ResNet18 | No | Adam | CE | 0.7050 | 0.7050 | 0.923 | 0.364 | 0.667 | 0.746 |
| ResNet18 | Yes | SGD | CE | — | — | — | — | — | — |
| ResNet18 | No | SGD | CE | — | — | — | — | — | — |
| VGG16 | Yes | Adam | CE | — | — | — | — | — | — |
| VGG16 | No | Adam | CE | — | — | — | — | — | — |
| VGG16 | Yes | SGD | CE | — | — | — | — | — | — |
| VGG16 | No | SGD | CE | — | — | — | — | — | — |

Fill remaining rows by running the full training suite.

---

## Prediction

Load any saved checkpoint and classify a single image:

```bash
python predict.py \
    --image datasets/cinnamon/C5/C5_01.JPG \
    --model resnet18 \
    --checkpoint checkpoints/resnet18_adam_pretrained.pth
```

Example output:

```
===================================
 Image:           datasets/cinnamon/C5/C5_01.JPG
 Architecture:    resnet18
 Checkpoint:      checkpoints/resnet18_adam_pretrained.pth
 Predicted Grade: C5
===================================
```

---

## Analysis

### Architecture: ResNet18 vs VGG16

**ResNet18** is generally preferred for this task because:
- 11M parameters vs VGG16's 138M — far less risk of overfitting on a small dataset
- Skip connections enable stable gradient flow when training from scratch
- Faster training iteration (critical for rapid experimentation)

**VGG16** may achieve competitive accuracy with pretrained weights but requires more regularization (dropout) and longer training time. Without pretrained weights it is significantly harder to optimize on a limited dataset. If dataset size increases substantially, VGG16's representational capacity may become an advantage.

### Optimizer: SGD vs Adam

**Adam** converges faster and requires less learning rate tuning. For small datasets and fine-tuning, Adam with a low learning rate (`1e-4`) performs well out of the box and is the recommended default.

**SGD with momentum** can match or exceed Adam's final accuracy given proper scheduling (`StepLR`). It often generalizes better when training time is not a constraint, and is the standard choice in the transfer learning literature (e.g., ResNet paper). Requires more careful hyperparameter selection.

### Transfer Learning: Pretrained vs From Scratch

The results show a clear advantage for pretrained weights, particularly for minority classes (C4, C5). The ImageNet backbone provides robust feature representations even when cinnamon-specific training data is limited. Training from scratch converges more slowly and underperforms on visually ambiguous classes without substantially more data and augmentation.

**Recommendation:** Use `ResNet18 + Adam + pretrained=True` as the production baseline. It achieves the best observed accuracy (91.4%), trains quickly, and generalizes well across all four grades.
