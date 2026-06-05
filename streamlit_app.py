import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model dari Google Drive jika belum ada
file_id = "1AABBF6Zh23ONuTxfw2baOPjmnubGZbhu"

if not os.path.exists("my_image_classifier_model.h5"):
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        "my_image_classifier_model.h5",
        quiet=False
    )

# Load model
model = tf.keras.models.load_model(
    "my_image_classifier_model.h5"
)

# Nama kelas
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# Judul aplikasi
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

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)

    score = np.argmax(prediction)

    st.success(
        f"Hasil Prediksi: {class_names[score]}"
    )
