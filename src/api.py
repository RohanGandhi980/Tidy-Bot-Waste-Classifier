# api.py
from fastapi import FastAPI, UploadFile, File
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import io

MODEL_PATH = "models/best_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = image.load_img(io.BytesIO(contents), target_size=(224, 224))

    # preprocessing
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    pred_class = 1 if preds[0][0] > 0.5 else 0
    return {
        "class": idx_to_class[pred_class],
        "confidence": float(preds[0][0])
    }
