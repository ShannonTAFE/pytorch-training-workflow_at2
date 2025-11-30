#!/usr/bin/env python3
"""
train.py — Complete, reproducible PyTorch training workflow with Weights & Biases tracking.

Implements ALL required features:
- Reproducible training with seeds
- CIFAR-10 dataloaders with augmentation
- CNN model definition
- Device: CPU/MPS/CUDA
- Training + validation loops with accuracy & loss
- Test evaluation + confusion matrix
- Best-model checkpointing
- Weights & Biases experiment tracking
- Metrics logged every epoch
- CLI via argparse
- Model parameter count print
- First batch shapes printed (evidence requirement)
- Clear project-ready structure
"""

# ===============================================================
# Imports
# ===============================================================
import argparse
import os
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as T

import wandb


# ===============================================================
# Utilities: Reproducibility
# ===============================================================
def set_seed(seed: int):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===============================================================
# CNN Model (Simple Baseline)
# ===============================================================
class CIFAR10Net(nn.Module):
    """Simple CNN for CIFAR-10 classification."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_parameters(model):
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================================================
# Data Loading
# ===============================================================
def build_dataloaders(data_dir, batch_size, num_workers, seed, val_split=0.1):

    # Augmentations
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2626)),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2626)),
    ])

    # Train/val
    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)

    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len

    train_set, val_set = random_split(
        full_train,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    # Test set
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    # Dataloaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    # Evidence: Print first batch shapes
    xb, yb = next(iter(train_loader))
    print(f"[INFO] First batch images: {xb.shape}")
    print(f"[INFO] First batch labels: {yb.shape}")

    return train_loader, val_loader, test_loader


# ===============================================================
# Training / Validation / Test Loops
# ===============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total, correct = 0, 0
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        _, preds = outputs.max(1)

        correct += preds.eq(yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, correct = 0, 0
    running_loss = 0.0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        loss = loss_fn(outputs, yb)

        running_loss += loss.item() * xb.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(yb).sum().item()
        total += yb.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def compute_conf_mat(model, loader, device, num_classes=10):
    model.eval()
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        outputs = model(xb)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())

    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


# ===============================================================
# Save / Load Checkpoints
# ===============================================================
def save_checkpoint(path, model, optimizer, epoch, best_val_acc, config):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "config": config
    }
    torch.save(ckpt, path)
    print(f"[INFO] Checkpoint saved: {path}")


# ===============================================================
# CLI Argument Parser
# ===============================================================
def parse_args():
    p = argparse.ArgumentParser(description="PyTorch CIFAR-10 Trainer")

    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="outputs")

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)

    # W&B
    p.add_argument("--wandb-project", type=str, default="pytorch_cifar10_workflow")
    p.add_argument("--wandb-name", type=str, default="run")

    return p.parse_args()


# ===============================================================
# Main Training Entry Point
# ===============================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    # Create folders
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Initialize W&B
    # ----------------------------
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

    # ----------------------------
    # Device selection
    # ----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    # ----------------------------
    # Data
    # ----------------------------
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.seed,
        args.val_split,
    )

    # ----------------------------
    # Model / Optimizer
    # ----------------------------
    model = CIFAR10Net().to(device)
    print(f"[INFO] Model parameters: {count_parameters(model):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # ----------------------------
    # Training Loop
    # ----------------------------
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device)

        val_loss, val_acc = evaluate(
            model, val_loader, loss_fn, device)

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        # Best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                out_dir / "best_model.pth",
                model, optimizer, epoch, best_val_acc, vars(args)
            )

    # ===============================================================
    # Final TEST Evaluation
    # ===============================================================
    print("\n===== Final Evaluation (Test Set) =====")
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"[RESULT] Test Accuracy: {test_acc*100:.2f}%")

    # Confusion Matrix
    conf_mat = compute_conf_mat(model, test_loader, device)
    np.save(out_dir / "confusion_matrix.npy", conf_mat)
    print("[INFO] Confusion matrix saved.")

    wandb.log({
        "test_loss": test_loss,
        "test_acc": test_acc,
    })
    wandb.finish()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
