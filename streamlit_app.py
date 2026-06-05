import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Covid dari X-Ray",
    layout="centered",
    page_icon="🩻"
)

# Nama kelas
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# Load model
@st.cache_resource
def load_model():

    model_path = "my_image_classifier_model.h5"

    if not os.path.exists(model_path):

        file_id = "1AABBF6Zh23ONuTxfw2baOPjmnubGZbhu"

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
        model = tf.keras.models.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"Gagal load model: {e}")
        return None


# Prediksi gambar
def prediksi_gambar(image_pil, model):

    img = image_pil.resize((224, 224))

    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    return class_names[predicted_class], confidence


# Tampilan
st.title("🩻 Deteksi Covid dari X-Ray")

st.write(
    "Upload gambar X-Ray paru-paru untuk mendeteksi Covid, Normal, atau Viral Pneumonia menggunakan Artificial Intelligence."
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

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang Diupload",
        use_container_width=True
    )

    if st.button("Prediksi"):

        with st.spinner("Sedang menganalisis gambar..."):

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
