#!/usr/bin/env python
"""
train_mnist_simple.py
A minimal, configurable CNN training script for MNIST with a train/val split.
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DatasetType = datasets.MNIST | datasets.EMNIST

MEAN = 0.1307
STD = 0.3081
DEFAULT_EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

class SimpleCNN(nn.Module):
    """
    CNN with configurable number of conv blocks.

    Each block: Conv -> (BatchNorm) -> Activation -> Pool -> (Dropout2d).
    """

    def __init__(
        self,
        num_conv_layers: int = 2,
        base_channels: int = 16,
        channel_multiplier: int = 2,
        kernel_size: int = 5,
        use_batchnorm: bool = False,
        activation: str = "relu",
        pool: str = "max",
        conv_dropout: float = 0.0,
        dropout: float = 0.0,
        num_fc_layers: int = 1,
        fc_hidden_dim: int = 128,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if num_conv_layers < 1:
            raise ValueError("num_conv_layers must be >= 1")
        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")
        if channel_multiplier < 1:
            raise ValueError("channel_multiplier must be >= 1")
        if num_fc_layers < 0:
            raise ValueError("num_fc_layers must be >= 0")

        padding = kernel_size // 2  # keep size before pooling
        layers = []
        in_channels = 1
        out_channels = base_channels

        act = self._make_activation(activation)
        pool_layer = self._make_pool(pool)
        conv_drop = nn.Dropout2d(conv_dropout) if conv_dropout > 0 else nn.Identity()

        for _ in range(num_conv_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(act)
            layers.append(pool_layer)
            layers.append(conv_drop)
            in_channels = out_channels
            out_channels *= channel_multiplier

        self.features = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Infer flatten size with a dummy forward pass.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            flat_dim = self.features(dummy).view(1, -1).size(1)

        classifier = []
        in_dim = flat_dim
        for _ in range(num_fc_layers):
            classifier.append(nn.Linear(in_dim, fc_hidden_dim))
            classifier.append(nn.ReLU(inplace=True))
            classifier.append(self.dropout)
            in_dim = fc_hidden_dim
        classifier.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "leaky_relu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError("activation must be 'relu' or 'leaky_relu'")

    @staticmethod
    def _make_pool(name: str) -> nn.Module:
        if name == "max":
            return nn.MaxPool2d(2)
        if name == "avg":
            return nn.AvgPool2d(2)
        raise ValueError("pool must be 'max' or 'avg'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def emnist_fix_orientation() -> transforms.Lambda:
    # EMNIST images are rotated and mirrored in the raw files.
    return transforms.Lambda(lambda x: torch.flip(torch.rot90(x, 1, [1, 2]), [2]))


def build_transforms(use_augmentation: bool, *, emnist_fix: bool) -> transforms.Compose:
    pre = []
    if use_augmentation:
        pre.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), shear=(-8, 8)))

    post = []
    if emnist_fix:
        post.append(emnist_fix_orientation())

    if not use_augmentation:
        return transforms.Compose([
            *pre,
            transforms.ToTensor(),
            *post,
            transforms.Normalize((MEAN,), (STD,)),
        ])

    return transforms.Compose([
        *pre,
        transforms.ToTensor(),
        *post,
        transforms.Normalize((MEAN,), (STD,)),
    ])


def get_dataset(
    name: str,
    data_root: Path,
    train: bool,
    transform: transforms.Compose,
    *,
    emnist_url: str,
) -> DatasetType:
    if name == "mnist":
        return datasets.MNIST(root=data_root, train=train, download=True, transform=transform)
    if name == "emnist_bal":
        # Override the default URL to avoid NIST redirects that break downloads.
        datasets.EMNIST.url = emnist_url
        return datasets.EMNIST(
            root=data_root,
            split="balanced",
            train=train,
            download=True,
            transform=transform,
        )
    raise ValueError("dataset must be 'mnist' or 'emnist_bal'")


def split_train_val_indices(total_len: int, val_split: float, seed: int) -> tuple[list[int], list[int]]:
    if val_split <= 0:
        raise ValueError("val_split must be > 0 (fraction or count)")

    if 0 < val_split < 1:
        val_len = int(total_len * val_split)
    else:
        val_len = int(val_split)

    if val_len <= 0 or val_len >= total_len:
        raise ValueError("val_split results in empty train/val set")

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_len, generator=generator).tolist()
    val_indices = perm[:val_len]
    train_indices = perm[val_len:]
    return train_indices, val_indices


def build_datasets(
    dataset: str,
    data_root: Path,
    use_augmentation: bool,
    val_split: float,
    seed: int,
    emnist_url: str,
) -> tuple[Subset, Subset, DatasetType, int]:
    emnist_fix = dataset.startswith("emnist")
    train_tf = build_transforms(use_augmentation, emnist_fix=emnist_fix)
    base_tf = build_transforms(False, emnist_fix=emnist_fix)

    full_train_aug = get_dataset(dataset, data_root, True, train_tf, emnist_url=emnist_url)
    full_train_base = get_dataset(dataset, data_root, True, base_tf, emnist_url=emnist_url)
    test_ds = get_dataset(dataset, data_root, False, base_tf, emnist_url=emnist_url)

    train_idx, val_idx = split_train_val_indices(len(full_train_base), val_split, seed)
    train_ds = Subset(full_train_aug, train_idx)
    val_ds = Subset(full_train_base, val_idx)
    num_classes = 10 if dataset == "mnist" else 47
    return train_ds, val_ds, test_ds, num_classes


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, epoch: int, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "args": vars(args),
        },
        path,
    )


def resolve_log_path(log_csv: Path | None, run_name: str, sweep_mode: bool) -> Path | None:
    if log_csv is None:
        return None
    log_csv = Path(log_csv)
    if log_csv.suffix != ".csv":
        return log_csv / f"{run_name}.csv"
    if sweep_mode:
        return log_csv.with_name(f"{log_csv.stem}_{run_name}{log_csv.suffix}")
    return log_csv


def open_csv_logger(path: Path | None):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "val_loss",
            "val_acc",
            "gap",
        ],
    )
    writer.writeheader()
    return f, writer


def apply_overrides(base_args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    args = argparse.Namespace(**vars(base_args))
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f"Unknown override key: {key}")
        if key in {"data_root", "save_dir", "log_csv", "sweep"} and value is not None:
            value = Path(value)
        setattr(args, key, value)
    return args


def run_experiment(args: argparse.Namespace, *, sweep_mode: bool = False) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_ds, val_ds, test_ds, num_classes = build_datasets(
        dataset=args.dataset,
        data_root=args.data_root,
        use_augmentation=args.use_augmentation,
        val_split=args.val_split,
        seed=args.seed,
        emnist_url=args.emnist_url,
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size * 2)

    model = SimpleCNN(
        num_conv_layers=args.num_conv_layers,
        base_channels=args.base_channels,
        channel_multiplier=args.channel_multiplier,
        kernel_size=args.kernel_size,
        use_batchnorm=args.use_batchnorm,
        activation=args.activation,
        pool=args.pool,
        conv_dropout=args.conv_dropout,
        dropout=args.dropout,
        num_fc_layers=args.num_fc_layers,
        fc_hidden_dim=args.fc_hidden_dim,
        num_classes=num_classes,
    ).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    best_val = 0.0
    warned = False

    log_path = resolve_log_path(args.log_csv, args.run_name, sweep_mode)
    log_file, log_writer = open_csv_logger(log_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                val_loss += loss.item() * x.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_loss = val_loss / total
        val_acc = correct / total

        gap = train_acc - val_acc
        print(
            f"Epoch {epoch:02}/{args.epochs}  "
            f"loss: {train_loss:.4f}  "
            f"train_acc: {train_acc:.3%}  "
            f"val_loss: {val_loss:.4f}  "
            f"val_acc: {val_acc:.3%}  "
            f"gap: {gap:+.3%}"
        )

        if log_writer is not None:
            log_writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "gap": f"{gap:.6f}",
                }
            )

        if args.save_best and val_acc > best_val:
            best_val = val_acc
            ckpt_path = args.save_dir / f"{args.checkpoint_name}_best.pt"
            save_checkpoint(ckpt_path, model, args, epoch, val_acc)

        if not warned and gap > args.overfit_gap_warn:
            warned = True
            print(
                f"[overfit] Train/val gap {gap:.2%} exceeds "
                f"threshold {args.overfit_gap_warn:.2%}. "
                "Consider more regularization or fewer layers."
            )

    # Final test accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Test accuracy: {correct/total:.3%}  ({correct}/{total})")

    if args.save_last:
        ckpt_path = args.save_dir / f"{args.checkpoint_name}_last.pt"
        save_checkpoint(ckpt_path, model, args, args.epochs, val_acc)

    if log_file is not None:
        log_file.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a configurable CNN on MNIST")

    # Model hyperparameters
    parser.add_argument("--num-conv-layers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--channel-multiplier", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=5, choices=[3, 5])
    parser.add_argument("--use-batchnorm", action="store_true")
    parser.add_argument("--activation", choices=["relu", "leaky_relu"], default="relu")
    parser.add_argument("--pool", choices=["max", "avg"], default="max")
    parser.add_argument("--conv-dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout on FC layers")
    parser.add_argument("--num-fc-layers", type=int, default=1)
    parser.add_argument("--fc-hidden-dim", type=int, default=128)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-best", action="store_true", help="Save best val checkpoint")
    parser.add_argument("--save-last", action="store_true", help="Save final checkpoint")
    parser.add_argument("--checkpoint-name", type=str, default="mnist_cnn")
    parser.add_argument("--run-name", type=str, default="mnist_cnn")
    parser.add_argument("--log-csv", type=Path, default=None)
    parser.add_argument("--sweep", type=Path, default=None, help="JSON list of arg overrides")
    parser.add_argument("--overfit-gap-warn", type=float, default=0.05)

    # Data hyperparameters
    parser.add_argument("--dataset", choices=["mnist", "emnist_bal"], default="mnist")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of train set used for validation")
    parser.add_argument("--use-augmentation", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emnist-url", type=str, default=DEFAULT_EMNIST_URL)

    args = parser.parse_args()

    if args.dataset != "mnist":
        if args.run_name == "mnist_cnn":
            args.run_name = f"{args.dataset}_cnn"
        if args.checkpoint_name == "mnist_cnn":
            args.checkpoint_name = args.run_name

    if args.sweep is not None:
        sweep_data = json.loads(args.sweep.read_text())
        if not isinstance(sweep_data, list):
            raise ValueError("sweep JSON must be a list of override objects")
        for idx, overrides in enumerate(sweep_data, start=1):
            if not isinstance(overrides, dict):
                raise ValueError("Each sweep entry must be an object")
            run_args = apply_overrides(args, overrides)
            if "run_name" not in overrides:
                run_args.run_name = f"{args.run_name}_{idx:02d}"
            if "checkpoint_name" not in overrides:
                run_args.checkpoint_name = run_args.run_name
            run_experiment(run_args, sweep_mode=True)
    else:
        run_experiment(args, sweep_mode=False)


if __name__ == "__main__":
    main()
