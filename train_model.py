import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
DATA_DIR = 'raw_data'      # Assumes raw_data is in the same folder
MODEL_DIR = 'model'        # Folder to save the model
LABELS_DIR = 'labels'      # Folder to save the labels
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
EPOCHS = 25 # Increased epochs for better training

def train():
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Loading images from: {DATA_DIR}")

    # Create directories if they don't exist
    Path(MODEL_DIR).mkdir(exist_ok=True)
    Path(LABELS_DIR).mkdir(exist_ok=True)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,       # Increased augmentation
        width_shift_range=0.2,   # Increased augmentation
        height_shift_range=0.2,  # Increased augmentation
        zoom_range=0.2,          # Increased augmentation
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VAL_SPLIT
    )

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VAL_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        color_mode='rgb'
    )

    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        color_mode='rgb'
    )

    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)

    print(f"\nFound {train_generator.samples} training images belonging to {num_classes} classes.")
    print(f"Found {validation_generator.samples} validation images belonging to {num_classes} classes.")
    print(f"Class names: {class_names}")

    # --- Define Model ---
    print("\nDefining the model using Transfer Learning (MobileNetV2)...")
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    )
    base_model.trainable = False  # Freeze the base

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Train Model ---
    print("\n--- Starting Model Training ---")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    print("\n--- Model Training Complete ---")

    # --- Save Artifacts ---
    model_path = Path(MODEL_DIR) / "chess_piece_model.keras"
    labels_path = Path(LABELS_DIR) / "class_names.txt"

    model.save(model_path)
    print(f"\nSaved trained model to: {model_path}")
    
    with open(labels_path, 'w') as f:
        f.write("\n".join(class_names))
    print(f"Saved class names to: {labels_path}")

if __name__ == "__main__":
    # To run this script, activate your venv and run:
    # python train_model.py
    train()
