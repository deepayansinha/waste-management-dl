import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ==========================
# CONFIGURATION
# ==========================

DATASET_PATH = "dataset"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "saved_model/waste_classifier.keras"

# ==========================
# LOAD DATASET
# ==========================

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# ==========================
# CLASS NAMES
# ==========================

class_names = train_dataset.class_names

print("\nWaste Categories:")
for i, name in enumerate(class_names):
    print(f"{i+1}. {name}")

# ==========================
# PERFORMANCE OPTIMIZATION
# ==========================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================
# BUILD CNN MODEL
# ==========================

model = keras.Sequential([

    # Normalize pixel values (0-255 -> 0-1)
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),

    # First Convolution Block
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    # Second Convolution Block
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    # Third Convolution Block
    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(),

    # Flatten
    layers.Flatten(),

    # Dense Layer
    layers.Dense(128, activation="relu"),

    # Prevent Overfitting
    layers.Dropout(0.3),

    # Output Layer
    layers.Dense(len(class_names), activation="softmax")
])

# ==========================
# COMPILE MODEL
# ==========================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# MODEL SUMMARY
# ==========================

print("\nModel Summary\n")
model.summary()

# ==========================
# TRAIN MODEL
# ==========================

print("\nTraining Started...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)

print("\nTraining Completed!")

# ==========================
# SAVE MODEL
# ==========================

os.makedirs("saved_model", exist_ok=True)

model.save(MODEL_SAVE_PATH)

print(f"\nModel saved successfully at:\n{MODEL_SAVE_PATH}")

# ==========================
# PLOT ACCURACY & LOSS
# ==========================

plt.figure(figsize=(12,5))

# Accuracy Graph
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Loss Graph
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()

# ==========================
# FINAL MESSAGE
# ==========================

print("\n==========================================")
print(" Smart Waste Management Model is Ready!")
print(" Saved Model :", MODEL_SAVE_PATH)
print(" Waste Classes :", class_names)
print("==========================================")