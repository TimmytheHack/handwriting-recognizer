#!/usr/bin/env python
# app.py – Streamlit handwriting recogniser (digits OR letters)

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch, torchvision.transforms as T
import numpy as np
from PIL import Image, ImageFilter
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0] / "src"))
from data_utils import MEAN, STD

# --------------------------------------------------------------------- sidebar
st.set_page_config(page_title="Handwriting Recognizer", layout="centered")
st.sidebar.header("Canvas")
stroke_width = st.sidebar.slider("Pen width", 1, 25, 12)
mode = st.sidebar.radio("Model", ["Digits (MNIST)", "Letters + Digits (EMNIST-bal)"])

# allow clearing the canvas
if st.sidebar.button("Clear"):
    st.session_state.key = st.session_state.get("key", 0) + 1
    st.rerun()

# ------------------------------------------------------------------ main title
st.title("🖌️ Draw a character")

canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280, width=280,
    drawing_mode="freedraw",
    key=st.session_state.get("key", 0),
)

# ----------------------------------------------------------- model + transform
@st.cache_resource(show_spinner=False)
def load_model(which: str):
    if which == "Digits (MNIST)":
        from models import LeNet5
        net = LeNet5()
        ckpt = "models/lenet_mnist_v1.pt"
        classes = [str(i) for i in range(10)]

    else:  # EMNIST-balanced
        from torchvision.models import resnet18
        import torch.nn as nn

        net = resnet18(num_classes=47, weights=None)
        net.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        ckpt = "models/resnet18_emnist_balanced.pt"

        # --- fixed 47-label order used during training --------------------
        classes = [
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","I","J",
            "K","L","M","N","O","P","Q","R","S","T",
            "U","V","W","X","Y","Z",
            "a","b","d","e","f","g","h","n","q","r","t"
        ]
        # index 0-46 now match the trained network’s logits exactly

    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net, classes


model, class_names = load_model(mode)

to_tensor = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((MEAN,), (STD,)),
])

# ---------- better crop → 28×28 tensor -------------------------------------
def preprocess(img: Image.Image) -> torch.Tensor | None:
    """
    1. Binarise -> bbox  2. Pad 12px margin  3. Erode 3×3  4. Resize+norm
    """
    # 1) binarise & tight crop
    bw = np.array(img) > 80               # stricter threshold
    if not bw.any():
        return None
    ys, xs = np.where(bw)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    glyph = img.crop((x0, y0, x1 + 1, y1 + 1))

    # 2) pad to centred square with +12px border
    w, h = glyph.size                     # PIL size = (w, h)
    side = max(w, h) + 12                 # generous margin
    square = Image.new("L", (side, side), 0)
    square.paste(glyph,
                 ((side - w) // 2,        # x-offset
                  (side - h) // 2))       # y-offset

    # 3) thin thick strokes (1-pixel each side)
    import cv2
    sq_np = np.array(square)
    sq_np = cv2.erode(sq_np,
                  kernel=np.ones((3, 3), np.uint8),
                  iterations=1)
    square = Image.fromarray(sq_np)
    # 4) resize → tensor → normalise
    tensor = to_tensor(square).unsqueeze(0)   # (1, 1, 28, 28)
    return tensor

# ---------- inference block -------------------------------------------------
if canvas.image_data is not None and canvas.image_data.sum() > 0:
    img = Image.fromarray(canvas.image_data.astype("uint8")[:, :, :3]).convert("L")
    tensor = preprocess(img)

    if tensor is None:                       # nothing drawn
        st.info("Draw something to get a prediction!")
        st.stop()

    # ── preview the 28×28 that the net sees ────────────────────────────────
    img_np = (tensor.squeeze()              # (28,28) tensor
                     .mul(STD)              # de-normalise
                     .add(MEAN)
                     .clamp(0, 1)
                     .cpu()
                     .numpy())              # → NumPy
    st.image(img_np, width=140)

    with torch.no_grad():
        prob = torch.softmax(model(tensor), dim=1).squeeze()

    topk = prob.topk(3)
    st.markdown(
        f"### **Top prediction → {class_names[topk.indices[0]]}** "
        f"({topk.values[0]:.1%})"
    )
    st.write({class_names[i]: f"{p:.1%}" for p, i in zip(topk.values, topk.indices)})
else:
    st.info("Draw something to get a prediction!")