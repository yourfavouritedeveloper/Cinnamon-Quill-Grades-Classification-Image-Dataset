

import os
import copy
import itertools
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

from models.cinnamon_model import get_model


DATA_DIR    = "datasets/cinnamon"
CKPT_DIR    = "checkpoints"
CLASSES     = ["Alba", "C4", "C5", "C5 Special"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE    = 224
BATCH_SIZE  = 8
EPOCHS      = 15
SEED        = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}\n")

os.makedirs(CKPT_DIR, exist_ok=True)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])



def load_datasets(data_dir: str):

    full_dataset = datasets.ImageFolder(data_dir)
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    torch.manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    train_ds.dataset = copy.deepcopy(full_dataset)
    train_ds.dataset.transform = train_transform

    val_copy = copy.deepcopy(full_dataset)
    val_copy.transform = eval_transform
    val_ds = torch.utils.data.Subset(val_copy, val_ds.indices)

    test_copy = copy.deepcopy(full_dataset)
    test_copy.transform = eval_transform
    test_ds = torch.utils.data.Subset(test_copy, test_ds.indices)

    targets = [full_dataset.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets, minlength=NUM_CLASSES).astype(float)
    weights = 1.0 / class_counts
    sample_weights = torch.tensor([weights[t] for t in targets], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total   += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / total
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    classwise_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return avg_loss, accuracy, f1, classwise_acc



def run_experiment(
    arch: str,
    optimizer_name: str,
    pretrained: bool,
    train_loader,
    val_loader,
    test_loader,
) -> dict:

    tag = f"{arch}_{optimizer_name}_{'pretrained' if pretrained else 'scratch'}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    model = get_model(arch, num_classes=NUM_CLASSES, pretrained=pretrained).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    writer = SummaryWriter(log_dir=f"runs/{tag}")

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, val_f1, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    elapsed = time.time() - t0
    writer.close()

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, classwise_acc = evaluate(model, test_loader, criterion, DEVICE)

    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pth")
    torch.save({
        "arch":       arch,
        "pretrained": pretrained,
        "optimizer":  optimizer_name,
        "state_dict": model.state_dict(),
        "test_acc":   test_acc,
        "test_f1":    test_f1,
    }, ckpt_path)

    print(f"\n  ✓ Test Accuracy: {test_acc:.4f}  F1: {test_f1:.4f}  Time: {elapsed:.0f}s")
    print(f"  ✓ Saved: {ckpt_path}")

    return {
        "arch":        arch,
        "pretrained":  pretrained,
        "optimizer":   optimizer_name,
        "test_loss":   test_loss,
        "test_acc":    test_acc,
        "test_f1":     test_f1,
        "classwise":   classwise_acc,
    }



def print_summary(results: list[dict]):
    header = (
        f"{'Model':<10} {'Pretrained':<12} {'Optimizer':<10} "
        f"{'Loss':<6} {'Accuracy':<10} {'F1':<8} "
        + "  ".join(f"{c:<12}" for c in CLASSES)
    )
    print("\n" + "=" * len(header))
    print("  EXPERIMENT SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        cw = "  ".join(f"{acc:<12.4f}" for acc in r["classwise"])
        print(
            f"{r['arch']:<10} {'Yes' if r['pretrained'] else 'No':<12} "
            f"{r['optimizer']:<10} {'CE':<6} {r['test_acc']:<10.4f} "
            f"{r['test_f1']:<8.4f} {cw}"
        )
    print("=" * len(header))



def main():
    print(f"Loading dataset from: {DATA_DIR}")
    train_loader, val_loader, test_loader = load_datasets(DATA_DIR)

    experiments = list(itertools.product(
        ["resnet18", "vgg16"],  
        ["sgd", "adam"],         
        [True, False],           
    ))

    results = []
    for arch, opt, pretrained in experiments:
        result = run_experiment(arch, opt, pretrained, train_loader, val_loader, test_loader)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
