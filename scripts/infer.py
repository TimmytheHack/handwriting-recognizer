import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

from src.models import LeNet5
from src.data_utils import MEAN, STD       # reuse canonical constants

# --- CLI ------------------------------------------------------------
parser = argparse.ArgumentParser(description="Predict a single digit image")
parser.add_argument("img", type=Path, help="Path to a 28×28 PNG/JPG")
args = parser.parse_args()

# --- Pre-processing pipeline (identical to training) ----------------
prep = transforms.Compose([
    transforms.Grayscale(),                # force 1-channel
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,))
])

# --- Load and preprocess -------------------------------------------
img = prep(Image.open(args.img).convert("RGB")).unsqueeze(0)   # shape (1,1,28,28)

# --- Load checkpoint & predict -------------------------------------
model = LeNet5()
model.load_state_dict(torch.load("models/lenet_mnist.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    probs = torch.softmax(model(img), dim=1).squeeze()
    pred  = probs.argmax().item()

print(f"Predicted digit: {pred}   (confidence {probs[pred]:.2%})")
