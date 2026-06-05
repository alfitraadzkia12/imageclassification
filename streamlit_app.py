import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Deteksi Covid dari X-Ray",
    page_icon="🩻",
    layout="centered"
)

# =========================
# LABEL KELAS
# =========================
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():

    model_path = "my_image_classifier_model.h5"

    if not os.path.exists(model_path):

        file_id = "1DEgQPFAyr-iUYbRDvbOm6e4Sm2vGR0lD"

        try:
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                model_path,
                quiet=False
            )

        except Exception as e:
            st.error(f"Gagal download model: {e}")
            return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None

# =========================
# PREDIKSI
# =========================
def prediksi_gambar(image, model):

    image = image.convert("RGB")

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array.astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)

    confidence = float(
        np.max(prediction) * 100
    )

    return (
        class_names[predicted_class],
        confidence
    )

# =========================
# UI
# =========================
st.title("🩻 Deteksi Covid dari X-Ray")

st.write(
    "Upload gambar X-Ray paru-paru untuk mendeteksi Covid, Normal, atau Viral Pneumonia."
)

with st.spinner("Menyiapkan model AI..."):
    model = load_model()

if model is None:
    st.stop()

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

        with st.spinner(
            "Sedang menganalisis gambar..."
        ):

            hasil, confidence = prediksi_gambar(
                image,
                model
            )

        st.success(
            f"Hasil Prediksi: {hasil}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )
