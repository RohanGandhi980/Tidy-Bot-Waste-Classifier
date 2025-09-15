import io, json, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

IMG_SIZE = 224
MODEL_PATH = "models/tidybot_best.h5"
CLASS_MAP_PATH = "models/class_indices.json"

app = FastAPI(title="TidyBot Classifier")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_MAP_PATH) as f:
    cls = json.load(f)
id2name = {v:k for k,v in cls.items()}

def preprocess_bytes(b):
    img = Image.open(io.BytesIO(b)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, 0)

@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    b = await file.read()
    x = preprocess_bytes(b)
    probs = model.predict(x, verbose=0)[0]
    idx = probs.argsort()[::-1][:topk]
    results = [{"label": id2name[i], "score": float(probs[i])} for i in idx]
    return {"topk": results}
