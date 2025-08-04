#!/usr/bin/env python
"""
train.py - training loop for MNIST *or* EMNIST-balanced.
"""

import argparse
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.datasets import MNIST
from src.data_utils import (
    get_train_val_datasets,
    emnist_balanced_loaders_split,
)
from torchvision.models import resnet18
import torch.nn as nn


ROOT = Path("data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin = torch.cuda.is_available()          # toggle for DataLoader

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "emnist_bal"],
        help="Training set to use",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw",
                        help="adamw (default) or classic sgd+momentum")

    args = parser.parse_args()

    if args.dataset == "mnist":
        # cached helper that returns an 80/20 split with your canonical aug
        train_ds, val_ds = get_train_val_datasets(ROOT, use_aug=True)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2)
        num_classes = 10
        ckpt_path = Path("models/lenet_mnist_v1.pt")
    else:  # EMNIST-balanced (47 classes)
        train_dl, val_dl, test_dl, class_names = emnist_balanced_loaders_split(
            data_dir=ROOT,
            batch_size=args.batch_size,
            num_workers=4,
            train_aug=None,
            pin_memory=pin, 
        )
        num_classes = len(class_names)          # 47
        ckpt_path = Path("models/resnet18_emnist_balanced.pt")

    

    model = resnet18(num_classes=num_classes, weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()        # remove the 2× down-sample
    model.to(device)


    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:  # adamw (default)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # ---------- train ----------
        model.train()
        running_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / len(train_dl.dataset)

        # ---------- validation ----------
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        scheduler.step()

        print(
            f"Epoch {epoch:02}/{args.epochs}  "
            f"loss: {avg_loss:.4f}  "
            f"val_acc: {val_acc:.3%}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"    saved new best model → {ckpt_path}  ({best_acc:.3%})")


if __name__ == "__main__":
    main()
