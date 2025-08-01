"""
data_utils.py
Step 4.1 - canonical transforms
"""
import torch
from torchvision import transforms

# Global constants – these stay the same forever
IMAGE_SIZE = (28, 28)
MEAN = 0.1307   # global MNIST mean (already in torchvision docs)
STD  = 0.3081   # global MNIST std

# --------------------------------------------------------------------------- #
#    T R A N S F O R M S
# --------------------------------------------------------------------------- #
BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),              # (H,W,C) PIL 0-255 → (C,H,W) float in [0,1]
    transforms.Normalize((MEAN,), (STD,))
])

AUG_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomAffine(
        degrees=12,
        translate=(2/28, 2/28)  # up to ±2 px shift
    ),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])


from torch.utils.data import random_split
from torchvision.datasets import MNIST
from pathlib import Path

def get_train_val_datasets(root: Path, val_size: int = 5_000, *, use_aug=False):
    """
    Return (train_ds, val_ds) with a fixed random seed so the split
    is *identical* every time we run the pipeline.
    """
    full_train = MNIST(
        root=root,
        train=True,
        transform=AUG_TRAIN_TRANSFORM if use_aug else BASE_TRANSFORM
    )
    assert len(full_train) == 60_000, "Unexpected MNIST train size"

    generator = torch.Generator().manual_seed(42)  # deterministic
    train_len = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_len, val_size],
                                    generator=generator)
    return train_ds, val_ds
