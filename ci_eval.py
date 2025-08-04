#!/usr/bin/env python
"""
ci_eval.py – run accuracy check in GitHub Actions.
Assumes the training step has just finished and left a model file on disk.
"""

import subprocess, sys, pathlib

# Resolve repo root so the script works from any CWD
ROOT = pathlib.Path(__file__).resolve().parents[0]

# Which dataset are we testing?  Read from the matrix or default to MNIST.
dataset = sys.argv[1] if len(sys.argv) > 1 else "mnist"

# Map dataset → checkpoint path produced by the 1-epoch smoke run
ckpt_map = {
    "mnist": ROOT / "models" / "lenet_mnist_v1.pt",
    "emnist_bal": ROOT / "models" / "resnet18_emnist_balanced.pt",
}

ckpt = ckpt_map[dataset]

print(f"[ci_eval] dataset={dataset}  ckpt={ckpt}")

# Call eval_test.py with explicit flags
subprocess.check_call(
    [
        sys.executable,
        "scripts/eval_test.py",
        "--ckpt", str(ckpt),
        "--dataset", dataset,
    ]
)
