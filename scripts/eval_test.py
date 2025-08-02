import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from src.data_utils import BASE_TRANSFORM
from src.models import LeNet5   # or keep a single model import

ROOT = Path("data")
ckpt = torch.load("models/lenet_mnist_v1.pt", map_location="cpu")

model = LeNet5()
model.load_state_dict(ckpt)
model.eval()

test_ds = MNIST(ROOT, train=False, transform=BASE_TRANSFORM)
test_dl = DataLoader(test_ds, batch_size=256)

correct = total = 0
with torch.no_grad():
    for x, y in test_dl:
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)

print(f"Test accuracy: {correct/total:.3%}  ({correct}/{total})")
