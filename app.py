import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

# ===============================
# STEP 0: Download model weights from Google Drive
# ===============================
MODEL_URL = "https://drive.google.com/uc?id=1KWMAhPJL2ft_UFqgcFoqCdcWogmnZldo"
MODEL_PATH = "pepper_classifier_model.keras"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model weights...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ===============================
# STEP 1: Rebuild the Model
# ===============================
def build_model(num_classes=2, img_size=224):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# ===============================
# STEP 2: Load Weights
# ===============================
@st.cache_resource
def load_model_safely():
    model = build_model(num_classes=2, img_size=224)
    try:
        model.load_weights(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load weights: {e}")
    return model

model = load_model_safely()
class_names = ["Bell Pepper", "Chili Pepper"]

# ===============================
# STEP 3: Preprocess Image
# ===============================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# ===============================
# STEP 4: Streamlit UI
# ===============================
st.title("🌶️ Pepper Classifier (Bell vs Chili)")
st.write("Upload an image of a bell pepper or chili pepper and the model will predict.")

uploaded_file = st.file_uploader("Upload a pepper image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if model:
        img_array = preprocess_image(img)
        preds = model.predict(img_array)
        result = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        st.subheader(f"🔮 Prediction: **{result}**")
        st.write(f"Confidence: {confidence:.2f}%")
