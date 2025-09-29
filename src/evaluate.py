# evaluate.py
import os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH_SIZE = 32
MODEL_PATH = "models/best_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"

# Load model & class indices
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)
idx_to_class = {v:k for k,v in class_indices.items()}

datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    "data/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# Predictions
preds = model.predict(val_gen)
y_true = val_gen.classes
y_pred = (preds > 0.5).astype(int)

print(classification_report(y_true, y_pred, target_names=list(class_indices.keys())))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(class_indices.keys()),
            yticklabels=list(class_indices.keys()))
plt.title("Confusion Matrix")
plt.savefig("models/confusion_matrix.png")
plt.close()
print("✅ Confusion matrix saved at models/confusion_matrix.png")
