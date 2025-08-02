#!/usr/bin/env python
"""
models.py  -  Step 5.1
Defines a tiny LeNet-style CNN for MNIST.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    Two conv layers + two FC layers  →  10-class logits.
    Param count ≈ 120 k — small enough for CPU training in minutes.
    """
    def __init__(self):
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # 28→24
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 12→8
        # Classifier
        self.fc1   = nn.Linear(16 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 128)
        self.fc3   = nn.Linear(128, 10)

    def forward(self, x):
        # (N,1,28,28) → logits (N,10)
        x = F.avg_pool2d(F.relu(self.conv1(x)), 2)   # 24→12
        x = F.avg_pool2d(F.relu(self.conv2(x)), 2)   # 8 →4
        x = torch.flatten(x, 1)                      # (N,256)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
