# Cinnamon Quill Grade Classification (PyTorch)

This project trains deep learning models to classify cinnamon quill images into four quality grades:

- **Alba**
- **C4**
- **C5**
- **C5 Special**

The workflow includes preprocessing, augmentation, two CNN architectures, transfer learning experiments, TensorBoard monitoring, evaluation, and saving trained models in `.pth` format.

---

## Project Structure

```bash
project/
│── datasets/
│ └── cinnamon/
│ ├── Alba/
│ ├── C4/
│ ├── C5/
│ └── C5 Special/
│
│── checkpoints/
│ ├── saved_model_pretrained.pth
│ └── saved_model_scratch.pth
│
│── utils.py
│── train.py
│── test.py
│── predict.py
│── README.md
```

---

## 1. Dataset Preparation

### Download dataset  
Place cinnamon images in labeled class folders.

### Preprocessing  
All images are resized and normalized:

- Resize to **224×224**
- Normalize using ImageNet mean/std
- Convert to tensors

### Data Augmentation  
Used during training:

- Random rotations  
- Horizontal & vertical flips  
- Random cropping  
- Optional color jitter  

These help models generalize better and deal with class imbalance.

---

## 2. Dataset Splitting

Dataset is split using:

- **70% Training**
- **15% Validation**
- **15% Test**

Using `random_split` from PyTorch.

---

## 3. CNN Architectures

Two network architectures were created:

### **ResNet18**
- With transfer learning
- Without transfer learning (trained from scratch)

### **VGG16**
- With transfer learning
- Without transfer learning

Toggle transfer learning:

```python
use_pretrained = True    # Transfer learning
use_pretrained = False   # Train from scratch
```

## Final Classifier

The final classifier layers are adapted for **4 output classes**: Alba, C4, C5, and C5 Special.

---

## 4. Training

Both **ResNet18** and **VGG16** models were trained using the following configurations:

### **Optimizers**
- SGD  
- Adam  

### **Loss Function**
- CrossEntropyLoss  

### **Additional Training Details**
- WeightedRandomSampler to reduce class imbalance  
- Batch size = 8  
- Learning rate scheduler  
- Dropout added to VGG classifier  
- Training and validation logs tracked in TensorBoard:

```bash
tensorboard --logdir=runs
```

## 5. Evaluation

Models are evaluated on the **test set** using:

- Test accuracy  
- Test loss  
- F1-score  
- Class-wise accuracy  
- Confusion matrix  

A final summary table is generated:

| Model     | Optimizer | Loss | Accuracy | F1  | Classwise Accuracy |
|-----------|-----------|------|----------|-----|---------------------|
| ResNet18  | SGD       | ...  | ...      | ... | ...                 |
| ResNet18  | Adam      | ...  | ...      | ... | ...                 |
| VGG16     | SGD       | ...  | ...      | ... | ...                 |
| VGG16     | Adam      | ...  | ...      | ... | ...                 |

---

## 6. Saving Models

Each experiment saves a `.pth` model checkpoint:

- `saved_model_pretrained.pth`  
- `saved_model_scratch.pth`  

Saved using:

```python
torch.save({
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict()
}, "checkpoints/model_name.pth")
```

## 7. Prediction Script

A separate `predict.py` script loads a trained `.pth` model and predicts the class of a single input image.

Run it with:

```bash
python predict.py
```

### Example Output
```bash
Image: datasets/cinnamon/C5/C5 01.JPG
Predicted Grade: C5
```

---

## Completed Requirements Checklist

- [x] Download and preprocess dataset  
- [x] Data augmentation  
- [x] Train/Val/Test split  
- [x] ResNet18 + VGG16 implemented  
- [x] Transfer learning ON/OFF  
- [x] Train using SGD & Adam  
- [x] TensorBoard logging  
- [x] Test accuracy and metrics  
- [x] Save model in `.pth` format  
- [x] Prediction script  

---

## Notes

- Imbalanced data may require `WeightedRandomSampler`.  
- Transfer learning typically increases performance.  
- Include TensorBoard **accuracy / loss** plots in your report.

---

## Final Goal

Build a reliable cinnamon grading model using deep learning and analyze how:

- **architecture choice** (ResNet vs. VGG), and  
- **training strategy** (transfer learning vs. training from scratch)

affect final performance.
