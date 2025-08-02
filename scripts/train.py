import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#!/usr/bin/env python
"""
train.py  -  Step 5.3
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.datasets import MNIST

from src.data_utils import get_train_val_datasets, BASE_TRANSFORM
from src.models import LeNet5

ROOT = Path("data")
NUM_EPOCHS = 10          # ❶ run for ten passes


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean()

def main():
    train_ds, val_ds = get_train_val_datasets(ROOT, use_aug=False)
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=256)

    model = LeNet5()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, NUM_EPOCHS + 1):
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

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
        val_acc = correct / total

        print(f"Epoch {epoch:02}/{NUM_EPOCHS}  "
            f"loss: {avg_loss:.4f}  "
            f"val_acc: {val_acc:.3%}")

if __name__ == "__main__":
    main()
