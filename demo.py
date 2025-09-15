# demo.py
import json, numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="TidyBot Demo", layout="centered")
st.title("🧹 TidyBot – Waste Classifier")

IMG_SIZE = 224
MODEL_PATH = "models/tidybot_best.h5"
CLASS_MAP_PATH = "models/class_indices.json"

@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_MAP_PATH) as f:
        cls = json.load(f)
    id2name = {v: k for k, v in cls.items()}
    return model, id2name

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, 0)

model, id2name = load_model_and_classes()

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if file:
    
    img = Image.open(file)
    st.image(img, caption="Input Image", use_column_width=True)

    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    idx = probs.argsort()[::-1]  

    best = int(idx[0])
    best_label = id2name[best]
    best_conf = float(probs[best])
    st.markdown(
        f"##Prediction: **{best_label}** ({best_conf:.2%})"
    )

    st.subheader("Top 3 Predictions")
    for i in idx[:3]:
        label = id2name[int(i)]
        conf = float(probs[i])
        st.write(f"**{label}**: {conf:.4f}")
        st.progress(min(1.0, conf))
else:
    st.info("Upload an image to see predictions.")
