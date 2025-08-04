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
        degrees=10,
        translate=(0.15, 0.15),
        shear=(-8, 8)
    ),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
    transforms.RandomErasing(p=0.05, scale=(0.02, 0.12))
])



from torch.utils.data import random_split, DataLoader
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


# --------------------------------------------------------------------------- #
#    E  M  N  I  S  T    B  A  L  A  N  C  E  D
# --------------------------------------------------------------------------- #
from torchvision import datasets
from torch.utils.data import DataLoader

def emnist_balanced_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    *,
    train_aug: transforms.Compose | None = None,
):
    """
    Return (train_loader, test_loader, class_names) for the EMNIST *balanced*
    split (47 classes: digits 0-9 + merged upper/lower letters).
    """
    train_tf = train_aug if train_aug is not None else AUG_TRAIN_TRANSFORM
    test_tf  = BASE_TRANSFORM

    train_ds = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=train_tf,
    )
    test_ds = datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
        transform=test_tf,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_ds.classes  # ➜ 47-item list

def emnist_balanced_loaders_split(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 4,
    val_ratio: float = 0.10,
    *,
    train_aug: transforms.Compose | None = None,
    pin_memory = True,
):
    """
    Returns (train_dl, val_dl, test_dl, class_names) where
      • train & val come from the original EMNIST-train set (shuffled)
      • val_ratio controls how much of that set becomes validation data
    """
    train_tf = train_aug if train_aug is not None else AUG_TRAIN_TRANSFORM
    test_tf  = BASE_TRANSFORM

    full_train = datasets.EMNIST(
        root=data_dir, split="balanced", train=True,
        download=True, transform=train_tf)

    # --- random 90 / 10 split --------------------------------------------
    val_len = int(len(full_train) * val_ratio)
    train_len = len(full_train) - val_len
    train_ds, val_ds = random_split(full_train, [train_len, val_len])

    test_ds = datasets.EMNIST(
        root=data_dir, split="balanced", train=False,
        download=True, transform=test_tf)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


    return train_dl, val_dl, test_dl, full_train.classes