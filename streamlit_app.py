import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# =========================
# DOWNLOAD MODEL
# =========================

FILE_ID = "1AABBF6Zh23ONuTxfw2baOPjmnubGZbhu"
MODEL_PATH = "my_image_classifier_model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Mengunduh model AI..."):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_PATH,
            quiet=False
        )

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# LABEL KELAS
# =========================

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# =========================
# TAMPILAN
# =========================

st.title("Deteksi Covid dari X-Ray")

uploaded_file = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang Diupload",
        use_container_width=True
    )

    if st.button("Prediksi"):

        # =========================
        # PREPROCESSING
        # =========================

        img = image.convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # =========================
        # PREDIKSI
        # =========================

        prediction = model.predict(img_array)

        st.write("Raw Prediction:", prediction)

        score = np.argmax(prediction)

        confidence = float(np.max(prediction) * 100)

        st.success(
            f"Hasil Prediksi: {class_names[score]}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )
