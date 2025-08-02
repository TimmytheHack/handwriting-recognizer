import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#!/usr/bin/env python
"""
train_once.py  -  Step 5.2
Runs ONE epoch on CPU to prove the full stack works.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST

from src.data_utils import get_train_val_datasets, BASE_TRANSFORM
from src.models import LeNet5

ROOT = Path("data")

def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean()

def main():
    # 1) datasets & loaders
    train_ds, val_ds = get_train_val_datasets(ROOT, use_aug=False)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)

    # 2) model, loss, optimiser
    model = LeNet5()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3) one training epoch
    model.train()
    running_loss = 0.0
    for x, y in train_dl:
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)

    avg_loss = running_loss / len(train_ds)
    print(f"Train loss: {avg_loss:.4f}")

    # 4) validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_dl:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

    print(f"Val accuracy: {correct/total:.3%}")

if __name__ == "__main__":
    main()
