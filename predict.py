import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ==========================
# CONFIGURATION
# ==========================

MODEL_PATH = "saved_model/waste_classifier.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMAGE_SIZE = (224, 224)

# ==========================
# LOAD MODEL
# ==========================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# ==========================
# GET IMAGE PATH
# ==========================

image_path = input("\nEnter image path: ").strip()

if not os.path.exists(image_path):
    print("Error: Image not found!")
    exit()

# ==========================
# LOAD & PREPROCESS IMAGE
# ==========================

img = image.load_img(image_path, target_size=IMAGE_SIZE)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

# ==========================
# PREDICT
# ==========================

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = CLASS_NAMES[predicted_index]

confidence = np.max(prediction) * 100

# ==========================
# DISPLAY RESULT
# ==========================

print("\n==============================")
print("Prediction Result")
print("==============================")
print(f"Waste Type : {predicted_class.capitalize()}")
print(f"Confidence : {confidence:.2f}%")
print("==============================")

# ==========================
# DISPOSAL SUGGESTIONS
# ==========================

tips = {
    "cardboard": "Recycle in the cardboard/paper recycling bin.",
    "glass": "Dispose in a glass recycling container.",
    "metal": "Place in a metal recycling bin.",
    "paper": "Recycle with dry paper waste.",
    "plastic": "Clean and recycle in the plastic recycling bin.",
    "trash": "Dispose as general waste."
}

print("\nRecommended Disposal:")
print(tips[predicted_class])