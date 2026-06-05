import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ======================
# LOAD MODEL
# ======================

@st.cache_resource
def load_model():

    model_path = "my_image_classifier_model.h5"

    if not os.path.exists(model_path):

        file_id = "1DEgQPFAyr-iUYbRDvbOm6e4Sm2vGR0lD"

        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            model_path,
            quiet=False
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model

model = load_model()

# ======================
# LABEL KELAS
# ======================

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# ======================
# UI
# ======================

st.title("☠️ Deteksi Covid dari X-Ray")

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

        img = image.resize((224, 224))

        img_array = np.array(img)

        img_array = img_array.astype(np.float32)

        img_array = img_array / 255.0

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        prediction = model.predict(img_array)

        # DEBUG
        st.subheader("DEBUG MODEL")

        st.write("Prediction Mentah:")
        st.write(prediction)

        st.write("Shape:")
        st.write(prediction.shape)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction) * 100

        st.success(
            f"Hasil Prediksi: {class_names[predicted_class]}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )
