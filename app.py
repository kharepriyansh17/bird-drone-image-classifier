import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Bird vs Drone Detector",
    layout="centered"
)

st.title("🦅 Bird vs 🚁 Drone Detection")
st.write("Confidence-based image classification")

# ----------------------------
# Load Model (Cached)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("notebooks/bird_drone_mobilenetv2.h5")

model = load_model()

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# Confidence-Based Prediction
# ----------------------------
raw_train = tf.keras.utils.image_dataset_from_directory(
    "/Users/priyanshkhare/Documents/bird-drone-project/data/classification_dataset/train",        # same path you used
    image_size=(160,160),
    batch_size=32,
    label_mode="binary"
)

CLASS_NAMES = raw_train.class_names
def predict_with_confidence(image, threshold=0.75):
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]   # sigmoid output

    if CLASS_NAMES[1] == "bird":
        bird_prob = pred
        drone_prob = 1 - pred
    else:
        drone_prob = pred
        bird_prob = 1 - pred

    if bird_prob >= threshold:
        label = "Bird"
        confidence = bird_prob
        status = "High confidence bird"
    elif drone_prob >= threshold:
        label = "Drone"
        confidence = drone_prob
        status = "High confidence drone"
    else:
        label = "Uncertain"
        confidence = max(bird_prob, drone_prob)
        status = "Low confidence – ambiguous"

    return label, confidence, bird_prob, drone_prob, status

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (Bird or Drone)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence, bird_p, drone_p, status = predict_with_confidence(image)

        st.subheader("🔍 Prediction Result")
        st.write(f"**Result:** {label}")
        st.write(f"**Status:** {status}")
        st.write(f"**Confidence:** `{confidence:.2f}`")

        st.markdown("---")
        st.subheader("📊 Class Probabilities")
        st.write(f"🦅 Bird: `{bird_p:.2f}`")
        st.write(f"🚁 Drone: `{drone_p:.2f}`")

        if confidence < 0.7:
            st.warning(
                "⚠️ Prediction confidence is low. Image may contain overlap or noise."
            )
