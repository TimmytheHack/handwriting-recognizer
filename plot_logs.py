#!/usr/bin/env python
"""
plot_logs.py
Plot training/validation curves from logs/run.csv.
"""

import argparse
import csv
from pathlib import Path


def read_metrics(path: Path) -> dict[str, list[float]]:
    data = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "gap": [],
    }
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["gap"].append(float(row["gap"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics from a CSV log file")
    parser.add_argument("--log", type=Path, default=Path("logs/run.csv"))
    parser.add_argument("--out", type=Path, default=Path("plots/run.png"))
    parser.add_argument("--show", action="store_true", help="Show plot window")
    args = parser.parse_args()

    try:
        import matplotlib
        if args.show:
            matplotlib.use("TkAgg")
        else:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from exc

    data = read_metrics(args.log)
    if not data["epoch"]:
        raise SystemExit(f"No rows found in {args.log}")

    best_idx = max(range(len(data["val_acc"])), key=lambda i: data["val_acc"][i])
    best_epoch = data["epoch"][best_idx]
    best_val = data["val_acc"][best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    fig.suptitle("Training Metrics")

    # Loss curves
    axes[0].plot(data["epoch"], data["train_loss"], label="train_loss")
    axes[0].plot(data["epoch"], data["val_loss"], label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(data["epoch"], data["train_acc"], label="train_acc")
    axes[1].plot(data["epoch"], data["val_acc"], label="val_acc")
    axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.0, 1.0)

    # Generalization gap
    axes[2].plot(data["epoch"], data["gap"], label="train_acc - val_acc")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("Gap")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out} (best val: {best_val:.3%} at epoch {best_epoch})")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
