import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import pandas as pd

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Deteksi Covid dari X-Ray",
    page_icon="🫁",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
    color: white;
}

.main-title{
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#cbd5e1;
    margin-bottom:30px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background-color:#1e293b;
    border:1px solid #475569;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    """
    <div class='main-title'>
    🫁 AI Covid Detection System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Deteksi Covid, Normal, dan Viral Pneumonia menggunakan citra X-Ray paru-paru berbasis Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# DOWNLOAD MODEL
# =========================

MODEL_NAME = "my_image_classifier_model.h5"

FILE_ID = "1GIM10m53KGe8pZHlPA7_gC9iMNgsJFjd"

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_NAME):

        with st.spinner("Mengunduh model AI..."):

            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_NAME,
                quiet=False
            )

    model = tf.keras.models.load_model(
        MODEL_NAME,
        compile=False
    )

    return model

model = load_model()

# =========================
# KELAS
# =========================

class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("📋 Informasi")

    st.write("""
    Sistem AI ini digunakan untuk:
    
    - Deteksi Covid
    - Deteksi Normal
    - Deteksi Viral Pneumonia
    
    Berdasarkan gambar X-Ray paru-paru.
    """)

    st.success("Model Accuracy ≈ 87.88%")

# =========================
# UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📤 Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    col1, col2 = st.columns([1,1])

    image = Image.open(uploaded_file).convert("RGB")

    with col1:

        st.image(
            image,
            caption="Gambar X-Ray",
            use_container_width=True
        )

    if st.button("🔍 Analisis X-Ray"):

        with st.spinner("AI sedang menganalisis gambar..."):

            img = image.resize((224,224))

            img_array = np.array(img)

            img_array = img_array.astype(np.float32)

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            prediction = model.predict(img_array)

            probs = tf.nn.softmax(
                prediction[0]
            ).numpy()

            predicted_class = np.argmax(probs)

            confidence = float(
                probs[predicted_class] * 100
            )

            hasil = class_names[predicted_class]

        with col2:

            st.subheader("📊 Hasil Analisis")

            if hasil == "Covid":

                st.error(
                    f"🔴 Prediksi: {hasil}"
                )

            elif hasil == "Normal":

                st.success(
                    f"🟢 Prediksi: {hasil}"
                )

            else:

                st.warning(
                    f"🟠 Prediksi: {hasil}"
                )

            st.write(
                f"### Confidence: {confidence:.2f}%"
            )

            st.progress(
                int(confidence)
            )

            st.subheader(
                "Probabilitas Semua Kelas"
            )

            df = pd.DataFrame(
                {
                    "Kelas": class_names,
                    "Probabilitas (%)":
                    np.round(
                        probs * 100,
                        2
                    )
                }
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.bar_chart(
                df.set_index("Kelas")
            )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "AI Covid Detection System | Deep Learning & Streamlit"
)
