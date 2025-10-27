import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import os
import matplotlib.pyplot as plt


# --- 1. Configuration ---
# This path is correct for Google Colab after unzipping
DATA_DIR = r'C:\Users\GFG19761\Desktop\img_to_fen\raw_data'
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_EPOCHS = 15 # How many times to go over the entire dataset. Start with 15-20.

# --- 2. Load Data (Same as before) ---
print(f"Loading images from: {DATA_DIR}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
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

# Get class info
class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

print(f"\nFound {train_generator.samples} training images belonging to {num_classes} classes.")
print(f"Found {validation_generator.samples} validation images belonging to {num_classes} classes.")
print(f"Class names: {class_names}")


# --- 3. Define the Model (Using Transfer Learning) ---
print("\nDefining the model using Transfer Learning (MobileNetV2)...")

# Load a powerful, pre-trained base model (MobileNetV2)
# We won't train this part, just use its learned "vision"
base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3), # (64, 64, 3)
    include_top=False, # Do NOT include its original 1000-class classifier
    weights='imagenet' # Use weights it learned from the giant ImageNet dataset
)

# Freeze the base model
base_model.trainable = False

# Add our own custom "head" (classifier) on top of the base model
inputs = Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x) # Pool the features
x = Dropout(0.2)(x) # Add dropout for regularization
# Our final layer: a Dense layer with 'softmax' activation
# It will have 'num_classes' outputs (one for each of your folders)
outputs = Dense(num_classes, activation='softmax')(x)

# Combine the base and the head into one model
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='categorical_crossentropy', # Good for multi-class classification
    metrics=['accuracy']
)

model.summary()


# --- 4. Train the Model ---
print("\n--- Starting Model Training ---")

# We will save the training history (accuracy, loss) to plot it later
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=NUM_EPOCHS
)

print("--- Model Training Complete ---")


# --- 5. Plot Training History ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(NUM_EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# --- 6. Save the Model ---
MODEL_SAVE_PATH = r'C:\Users\GFG19761\Desktop\img_to_fen\model\chess_piece_model.h5'
print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

# Save the class names (labels) to a text file
LABEL_SAVE_PATH = r'C:\Users\GFG19761\Desktop\img_to_fen\labels\class_names.txt'
with open(LABEL_SAVE_PATH, 'w') as f:
    for item in class_names:
        f.write(f"{item}\n")
print(f"Saving class names to: {LABEL_SAVE_PATH}")

print("\nAll done! You can now download 'chess_piece_model.h5' and 'class_names.txt' from the file browser on the left.")
