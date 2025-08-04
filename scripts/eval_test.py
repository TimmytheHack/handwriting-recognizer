#!/usr/bin/env python
"""
eval_test.py - one-line test-set accuracy checker for either dataset.
"""

import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.data_utils import BASE_TRANSFORM, emnist_balanced_loaders
from src.models import LeNet5


ROOT = Path("data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt weights file")
    parser.add_argument(
        "--dataset",
        default="mnist",
        choices=["mnist", "emnist_bal"],
        help="Which test set to run",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.dataset == "mnist":
        test_ds = MNIST(ROOT, train=False, download=True, transform=BASE_TRANSFORM)
        test_dl = DataLoader(test_ds, batch_size=args.batch_size)
        num_classes = 10
    else:  # EMNIST-balanced
        _, test_dl, class_names = emnist_balanced_loaders(
            data_dir=ROOT, batch_size=args.batch_size
        )
        num_classes = len(class_names)  # 47

    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Test accuracy: {correct/total:.3%}  ({correct}/{total})")


if __name__ == "__main__":
    main()
