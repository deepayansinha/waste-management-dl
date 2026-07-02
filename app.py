import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("♻️ Smart Waste Management")

st.sidebar.info("""
### Instructions

1. Upload an image of waste.
2. AI will classify it.
3. View confidence score.
4. Get recycling suggestions.
""")

st.sidebar.markdown("---")
st.sidebar.write("### Built With")
st.sidebar.write("✅ TensorFlow")
st.sidebar.write("✅ Streamlit")
st.sidebar.write("✅ CNN Deep Learning")

# ======================================================
# LOAD MODEL
# ======================================================

MODEL_PATH = os.path.join("saved_model", "waste_classifier.keras")

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found!\n\nExpected location:\n{MODEL_PATH}")
        st.stop()

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ======================================================
# CLASS NAMES
# ======================================================

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ======================================================
# RECYCLING TIPS
# ======================================================

tips = {

    "cardboard": {
        "bin": "📦 Cardboard Recycling Bin",
        "tip": "Flatten cardboard boxes before recycling."
    },

    "glass": {
        "bin": "🍾 Glass Recycling Bin",
        "tip": "Handle broken glass carefully."
    },

    "metal": {
        "bin": "🥫 Metal Recycling Bin",
        "tip": "Clean cans before recycling."
    },

    "paper": {
        "bin": "📄 Paper Recycling Bin",
        "tip": "Keep paper clean and dry."
    },

    "plastic": {
        "bin": "🧴 Plastic Recycling Bin",
        "tip": "Wash plastic containers before recycling."
    },

    "trash": {
        "bin": "🗑 General Waste Bin",
        "tip": "Dispose responsibly."
    }

}

# ======================================================
# ENVIRONMENTAL FACTS
# ======================================================

facts = {

    "cardboard":
    "Cardboard is biodegradable and highly recyclable.",

    "glass":
    "Glass can be recycled endlessly without losing quality.",

    "metal":
    "Recycling aluminum saves up to 95% of the energy needed to produce new aluminum.",

    "paper":
    "Recycling paper helps reduce deforestation.",

    "plastic":
    "Plastic can take hundreds of years to decompose.",

    "trash":
    "Reducing landfill waste helps protect the environment."

}

# ======================================================
# IMAGE PREPROCESSING
# ======================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image)

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ======================================================
# MAIN PAGE
# ======================================================

st.title("♻️ Smart Waste Management")

st.markdown("""
### Deep Learning Based Waste Classification

Upload an image of waste and let Artificial Intelligence identify it instantly.
""")

st.markdown("---")

uploaded_file = st.file_uploader(
    "📤 Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# PREDICTION
# ======================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing image..."):

        prediction = model.predict(processed_image, verbose=0)

    predicted_index = np.argmax(prediction)

    predicted = class_names[predicted_index]

    confidence = float(np.max(prediction))

    st.markdown("---")

    st.subheader("Prediction")

    st.success(f"Detected Waste: **{predicted.upper()}**")

    st.subheader("Confidence")

    st.progress(confidence)

    st.write(f"### {confidence*100:.2f}%")

    st.subheader("Recycling Suggestion")

    st.info(tips[predicted]["bin"])

    st.success(tips[predicted]["tip"])

    st.subheader("Environmental Fact")

    st.warning(facts[predicted])

    st.subheader("Prediction Probabilities")

    probabilities = pd.DataFrame({
        "Waste Type": class_names,
        "Confidence": prediction[0]
    })

    for i, name in enumerate(class_names):
        st.write(
            f"**{name.capitalize()}** : {prediction[0][i]*100:.2f}%"
        )

    st.subheader("Confidence Chart")

    st.bar_chart(
        probabilities.set_index("Waste Type")
    )

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.caption(
    "Developed by Tiny | Smart Waste Management Using Deep Learning | B.Tech AI & ML Project"
)