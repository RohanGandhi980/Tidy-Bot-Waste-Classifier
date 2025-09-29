# training.py
import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from preprocessing import create_datagen

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.json")


train_gen = datagen.flow_from_directory(
    "data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",   # only 2 classes
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    "data/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

base_model = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(224,224,3))

for layer in base_model.layers[:150]:
    layer.trainable = False
for layer in base_model.layers[150:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
preds = Dense(1, activation="sigmoid")(x)   

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr]
)

print(f"✅ Training complete. Best model saved at {MODEL_PATH}")

#saving the model
with open(CLASS_INDICES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f)
print(f"✅ Class indices saved at {CLASS_INDICES_PATH}")
