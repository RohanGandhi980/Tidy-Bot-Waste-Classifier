import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, callbacks
import os, json

# Paths
DATA_DIR = "dataset-split"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Base Model (Transfer Learning)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # freeze layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(train_data.num_classes, activation="softmax")
])

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Save class indices (for predictions later)
os.makedirs("models", exist_ok=True)
with open("models/class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# Callbacks
cbs = [
    callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
    callbacks.ModelCheckpoint("models/tidybot_best.h5", monitor="val_accuracy", save_best_only=True)
]

# Train
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=cbs)

# Save final model
model.save("models/tidybot_final.h5")
print("Training complete. Models saved in /models/")
