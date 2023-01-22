import streamlit as st
import torch
import src.config as CFG
from PIL import Image
import numpy as np
import json
from src.model import HNet
import time


def classify(model, file, mapping):
    image = Image.open(file).convert("RGB")
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)

    outputs = model(image)

    _, preds = torch.max(outputs, 1)

    st.info(
        f"The predicted character is: {mapping[str(preds[0].item())]}",
    )


st.markdown("<h1>Hindi Character Recognition<h1>", unsafe_allow_html=True)

option = st.sidebar.radio(
    label="Classify Hindi Digit or Vyanjan ?", options=["Digit", "Vyanjan"], index=0
)


def upload_and_classify(model, mapping):
    file = st.file_uploader("Upload image!")

    if file is not None:

        st.image(file, use_column_width=True)
        button = st.button("Predict")

        if button:
            with st.spinner("Classifying....."):
                classify(model=model, file=file, mapping=mapping)


mapping, model = None, None

if option == "Digit":
    if CFG.BEST_MODEL_DIGIT.exists():
        model = HNet(num_classes=10)
        model.load_state_dict(torch.load(CFG.BEST_MODEL_DIGIT, map_location=CFG.DEVICE))
        with open(CFG.INDEX_DIGIT, "r") as f:
            mapping = json.load(f)
        upload_and_classify(model, mapping)
    else:
        st.error("No model exists! First Train model!")
else:
    if CFG.BEST_MODEL_VYANJAN.exists():
        model = HNet(num_classes=36)
        model.load_state_dict(
            torch.load(CFG.BEST_MODEL_VYANJAN, map_location=CFG.DEVICE)
        )
        with open(CFG.INDEX_VYNAJAN, "r") as f:
            mapping = json.load(f)
        upload_and_classify(model, mapping)
    else:
        st.error("No model exists! First train model!")
