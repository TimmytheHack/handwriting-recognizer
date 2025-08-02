"""
ci_eval.py  - run test-set evaluation and exit non-zero if accuracy < 99 %
"""

import re, subprocess, sys, pathlib

# run the existing evaluator
result = subprocess.check_output(
    [sys.executable, "scripts/eval_test.py"], text=True
)
print(result)

match = re.search(r"([0-9]+\.[0-9]+)", result)
acc = float(match.group(1)) if match else 0.0
print(f"Parsed accuracy = {acc:.2f}%")

if acc < 99.0:
    sys.exit(f"CI failed: accuracy {acc:.2f}% is below 99.0%")
