#!/usr/bin/env python
"""
models.py  -  tiny LeNet-style CNN, now with a configurable head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """
    Two conv layers + two FC layers → logits.
    Default is 10 classes (MNIST), but pass `num_classes=47`
    for EMNIST-balanced or 26 for EMNIST-letters.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # feature extractor
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 28 → 24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 12 → 8
        # classifier
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)  # 24 → 12
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)  # 8  → 4
        x = torch.flatten(x, 1)                     # (N, 256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
