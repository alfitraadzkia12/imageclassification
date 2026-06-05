import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import os
import gdown

# ==================================
# CONFIG
# ==================================
st.set_page_config(
    page_title="Deteksi Covid X-Ray",
    page_icon="🩻",
    layout="wide"
)

# ==================================
# CSS
# ==================================
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

.main-title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:25px;
}

.result-card{
    background:#1e293b;
    padding:25px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(0,0,0,0.3);
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
}

</style>
""", unsafe_allow_html=True)

# ==================================
# DOWNLOAD MODEL DARI DRIVE
# ==================================
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):

    file_id = "1DEgQPFAyr-iUYbRDvbOm6e4Sm2vGR0lD"

    url = f"https://drive.google.com/uc?id={file_id}"

    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(
            url,
            MODEL_PATH,
            quiet=False
        )

# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==================================
# CLASS
# ==================================
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# ==================================
# PREPROCESS
# ==================================
def predict_image(image):

    image = image.convert("RGB")

    image = image.resize((224,224))

    img_array = np.array(image)

    img_array = img_array.astype(np.float32)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    probs = tf.nn.softmax(
        prediction[0]
    ).numpy()

    predicted_class = np.argmax(probs)

    confidence = probs[predicted_class] * 100

    return (
        class_names[predicted_class],
        confidence,
        probs
    )

# ==================================
# HEADER
# ==================================
st.markdown(
    """
    <div class="main-title">
    🩻 Deteksi Covid dari X-Ray
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Klasifikasi Covid, Normal, dan Viral Pneumonia menggunakan Deep Learning
    </div>
    """,
    unsafe_allow_html=True
)

# ==================================
# UPLOAD
# ==================================
uploaded_file = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Gambar X-Ray",
            use_container_width=True
        )

    with col2:

        label, confidence, probs = predict_image(image)

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown("## 📊 Hasil Analisis")

        if label == "Covid":
            st.error(
                f"🔴 Prediksi: {label}"
            )

        elif label == "Normal":
            st.success(
                f"🟢 Prediksi: {label}"
            )

        else:
            st.warning(
                f"🟡 Prediksi: {label}"
            )

        st.markdown(
            f"## Confidence: {confidence:.2f}%"
        )

        st.progress(
            float(confidence / 100)
        )

        if confidence >= 80:
            st.success(
                "Model sangat yakin terhadap prediksi."
            )

        elif confidence >= 60:
            st.info(
                "Model cukup yakin terhadap prediksi."
            )

        else:
            st.warning(
                "Tingkat keyakinan model masih rendah."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        # ==========================
        # TABEL
        # ==========================
        st.markdown(
            "### 📋 Probabilitas Semua Kelas"
        )

        df = pd.DataFrame({
            "Kelas": class_names,
            "Probabilitas (%)":
            np.round(
                probs * 100,
                2
            )
        })

        st.dataframe(
            df,
            use_container_width=True
        )

        # ==========================
        # CHART
        # ==========================
        fig = px.bar(
            df,
            x="Kelas",
            y="Probabilitas (%)",
            text="Probabilitas (%)",
            title="Distribusi Probabilitas Prediksi"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================
# FOOTER
# ==================================
st.markdown(
    """
    <div class="footer">
    Sistem Deteksi Covid, Normal, dan Viral Pneumonia menggunakan CNN dan Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
