# test.py
import os, json, numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

DATA_DIR = "dataset-split"
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_DIR = "models"

# ----- load model -----
model_path = os.path.join(MODEL_DIR, "tidybot_best.h5")
if not os.path.exists(model_path):
    model_path = os.path.join(MODEL_DIR, "tidybot_final.h5")
model = tf.keras.models.load_model(model_path)
print(f"Loaded model: {model_path}")

# ----- load class indices -----
with open(os.path.join(MODEL_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)
idx2class = {v: k for k, v in class_indices.items()}
class_names = [idx2class[i] for i in range(len(idx2class))]

# ----- test generator -----
test_gen = ImageDataGenerator(rescale=1./255)
test_ds = test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ----- eval -----
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest accuracy: {acc:.4f} | Test loss: {loss:.4f}")

# ----- predictions & metrics -----
probs = model.predict(test_ds, verbose=1)
y_pred = probs.argmax(axis=1)
y_true = test_ds.classes

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

cm = confusion_matrix(y_true, y_pred)
print("Confusion matrix (raw counts):\n", cm)

# ----- save artifacts -----
os.makedirs(MODEL_DIR, exist_ok=True)

# save report to txt
with open(os.path.join(MODEL_DIR, "test_classification_report.txt"), "w") as f:
    f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# plot & save confusion matrix
plt.figure(figsize=(8,6))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

print("\nSaved:")
print(f"- models/test_classification_report.txt")
print(f"- models/confusion_matrix.png")
