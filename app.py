
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model(
    "my_image_classifier_model.h5"
)

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

st.title("Deteksi Covid dari X-Ray")

uploaded_file = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    score = np.argmax(prediction)

    st.success(
        f"Hasil Prediksi: {class_names[score]}"
    )
