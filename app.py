#!/usr/bin/env python
# app.py – Streamlit digit recogniser (fixed black background, no colour picker)

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch, torchvision.transforms as T
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import pathlib, sys

# make local package importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0] / "src"))
from models import LeNet5
from data_utils import MEAN, STD

# -------------------------- Streamlit page ----------------------------------
st.set_page_config(page_title="MNIST Digit Recogniser", layout="centered")

# sidebar – only pen width and clear button now
st.sidebar.header("Canvas options")
stroke_width = st.sidebar.slider("Pen width", 1, 25, 12)
clear_button = st.sidebar.button("Clear canvas")

# force a fresh canvas on clear
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0
if clear_button:
    st.session_state.canvas_key += 1
    st.rerun()

# ---------------------------- main canvas -----------------------------------
st.title("🖌️ Draw a digit (0 – 9)")
canvas = st_canvas(
    fill_color="rgba(255,255,255,1)",          # white ink
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",                # fixed black background
    height=280, width=280,
    drawing_mode="freedraw",
    key=f"canvas{st.session_state.canvas_key}",
)

# --------------------------- model & transform ------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    net = LeNet5()
    net.load_state_dict(torch.load("models/lenet_mnist_v1.pt", map_location="cpu"))
    net.eval()
    return net

model = load_model()
to_tensor = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((MEAN,), (STD,)),
])

# ------------------------------- inference ----------------------------------
if canvas.image_data is not None and canvas.image_data.sum() > 0:
    # grab RGB, discard alpha
    img_rgb = Image.fromarray(canvas.image_data.astype("uint8")[:, :, :3])

    # build mask: anything brighter than 30 → stroke
    gray = img_rgb.convert("L")
    mask = gray.point(lambda p: 255 if p > 30 else 0)

    # bail out if nothing drawn
    if np.max(mask) == 0:
        st.info("Draw something to get a prediction!")
        st.stop()

    # optional: thicken strokes a tiny bit so very thin lines survive
    mask = mask.filter(ImageFilter.MaxFilter(3))

    ink = Image.new("L", img_rgb.size, 0)
    ink.paste(255, mask=mask)

    tensor = to_tensor(ink).unsqueeze(0)          # (1,1,28,28)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()
        pred  = int(probs.argmax())
        conf  = float(probs[pred])

    st.markdown(f"### Prediction : **{pred}**")
    st.progress(value=int(conf * 100), text=f"{conf:.1%} confidence")
else:
    st.info("Draw something to get a prediction!")
