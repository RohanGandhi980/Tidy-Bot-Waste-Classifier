from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "dataset-split/train",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    "dataset-split/val",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical"
)
