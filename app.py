#!/usr/bin/env python
"""
app.py
Streamlit live handwriting recognizer for MNIST and EMNIST-balanced.
"""

from pathlib import Path
import time

import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch

from train_mnist_simple import SimpleCNN, MEAN, STD

EMNIST_BAL_CLASSES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "a", "b", "d", "e", "f", "g", "h", "n", "q", "r", "t",
]


def shift_image(arr: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros_like(arr)

    if dx >= 0:
        src_x = slice(0, w - dx)
        dst_x = slice(dx, w)
    else:
        src_x = slice(-dx, w)
        dst_x = slice(0, w + dx)

    if dy >= 0:
        src_y = slice(0, h - dy)
        dst_y = slice(dy, h)
    else:
        src_y = slice(-dy, h)
        dst_y = slice(0, h + dy)

    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def preprocess(img: Image.Image, *, threshold: int) -> torch.Tensor | None:
    # Convert to grayscale and binarize.
    img = img.convert("L")
    arr = np.array(img)
    mask = arr > threshold
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    glyph = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Scale to 20x20 box while keeping aspect ratio.
    w, h = glyph.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    glyph = glyph.resize((new_w, new_h), Image.BILINEAR)

    # Paste into 28x28 canvas.
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - new_w) // 2, (28 - new_h) // 2)
    canvas.paste(glyph, offset)

    arr28 = np.array(canvas, dtype=np.float32) / 255.0
    total = arr28.sum()
    if total <= 1e-6:
        return None

    # Center-of-mass shift to match MNIST style.
    ys, xs = np.indices(arr28.shape)
    cx = float((xs * arr28).sum() / total)
    cy = float((ys * arr28).sum() / total)
    dx = int(round(14 - cx))
    dy = int(round(14 - cy))
    arr28 = shift_image(arr28, dx, dy)

    tensor = torch.from_numpy(arr28).unsqueeze(0)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0)


def resolve_checkpoint(mode: str, override: str | None) -> Path:
    if override:
        return Path(override)
    if mode == "MNIST":
        return Path("checkpoints/mnist_cnn_best.pt")
    return Path("checkpoints/emnist_bal_cnn_best.pt")


def build_model_from_args(args: dict, num_classes: int) -> SimpleCNN:
    defaults = dict(
        num_conv_layers=2,
        base_channels=16,
        channel_multiplier=2,
        kernel_size=5,
        use_batchnorm=False,
        activation="relu",
        pool="max",
        conv_dropout=0.0,
        dropout=0.0,
        num_fc_layers=1,
        fc_hidden_dim=128,
    )
    defaults.update({k: v for k, v in args.items() if k in defaults})
    return SimpleCNN(num_classes=num_classes, **defaults)


@st.cache_resource(show_spinner=False)
def load_model(ckpt_path: str, num_classes: int) -> SimpleCNN:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    model = build_model_from_args(args, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


st.set_page_config(page_title="Live Handwriting Recognizer", layout="centered")
st.title("Live Handwriting Recognizer")

st.sidebar.header("Model")
mode = st.sidebar.radio("Dataset", ["MNIST", "EMNIST-balanced"])
override_path = st.sidebar.text_input("Checkpoint path (optional)", "")
live_update = st.sidebar.checkbox("Live prediction", value=True)
throttle_ms = st.sidebar.slider("Update interval (ms)", 50, 500, 150, 10)

st.sidebar.header("Canvas")
stroke_width = st.sidebar.slider("Pen width", 1, 25, 12)
threshold = st.sidebar.slider("Threshold", 10, 200, 80, 5)
if st.sidebar.button("Clear"):
    st.session_state.key = st.session_state.get("key", 0) + 1
    st.rerun()

canvas_kwargs = dict(
    fill_color="rgba(255,255,255,1)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.get('key', 0)}",
)
try:
    import inspect
    if "realtime_update" in inspect.signature(st_canvas).parameters:
        canvas_kwargs["realtime_update"] = live_update
except (ValueError, TypeError):
    pass

canvas = st_canvas(**canvas_kwargs)

is_emnist = mode != "MNIST"
class_names = [str(i) for i in range(10)] if not is_emnist else EMNIST_BAL_CLASSES
ckpt_path = resolve_checkpoint("MNIST" if not is_emnist else "EMNIST", override_path.strip())

if not ckpt_path.exists():
    st.error(f"Checkpoint not found: {ckpt_path}")
    st.stop()

model = load_model(str(ckpt_path), num_classes=len(class_names))

if "last_pred_time" not in st.session_state:
    st.session_state.last_pred_time = 0.0

if canvas.image_data is not None and canvas.image_data.sum() > 0:
    now = time.time()
    if live_update and (now - st.session_state.last_pred_time) * 1000 < throttle_ms:
        st.info("Drawing... (throttled)")
        st.stop()
    st.session_state.last_pred_time = now

    img = Image.fromarray(canvas.image_data.astype("uint8")[:, :, :3])
    tensor = preprocess(img, threshold=threshold)
    if tensor is None:
        st.info("Draw something to get a prediction.")
        st.stop()

    # Preview the 28x28 input
    preview = tensor.squeeze(0).squeeze(0).mul(STD).add(MEAN).clamp(0, 1).numpy()
    st.image(preview, width=140)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
    topk = probs.topk(3)
    st.markdown(f"**Top prediction:** {class_names[topk.indices[0]]} ({topk.values[0]:.1%})")
    st.write({class_names[i]: f"{p:.1%}" for p, i in zip(topk.values.tolist(), topk.indices.tolist())})
else:
    st.info("Draw something to get a prediction.")
