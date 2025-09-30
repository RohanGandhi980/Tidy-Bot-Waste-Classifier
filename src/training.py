# training.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import json
from pathlib import Path

DATA_DIR = Path("data")
TRAIN_DIR, VAL_DIR = DATA_DIR / "train", DATA_DIR / "val"

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                   horizontal_flip=True, fill_mode='nearest')

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="binary")

# Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer=optimizers.Adam(1e-4),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2)
]

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

# Save class indices
class_indices = train_gen.class_indices
with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("✅ Training complete. Best model saved at models/best_model.h5")
