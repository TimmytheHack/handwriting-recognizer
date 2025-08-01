#!/usr/bin/env python
"""
download_mnist.py  -  Step 3.2
Accepts an optional --root path and prints the absolute version.
No downloading yet; we're just wiring up the CLI.
"""

import argparse
from pathlib import Path
import sys

def main() -> None:
    # --- CLI ---------------------------------------------------------------
    parser = argparse.ArgumentParser(description="MNIST downloader (step 3.3)")
    parser.add_argument("--root", default="data/",
                        help="Destination folder (default: data/)")
    args = parser.parse_args()

    # --- Resolve and create folder ----------------------------------------
    root_path = Path(args.root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {root_path}")

    # --- Download train & test splits -------------------------------------
    from torchvision.datasets import MNIST

    train_ds = MNIST(root=root_path, train=True,  download=True)
    test_ds  = MNIST(root=root_path, train=False, download=True)

    # --- Quick sanity summary ---------------------------------------------
    print(f"Train images: {len(train_ds):,}")
    print(f" Test images: {len(test_ds):,}")

    # Class histogram (0-9)
    from collections import Counter
    hist = Counter(train_ds.targets.tolist())
    print("Train distribution:", dict(sorted(hist.items())))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
