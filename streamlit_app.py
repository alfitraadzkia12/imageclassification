import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import gdown
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Covid X-Ray",
    page_icon="🩻",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:white;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
    margin-bottom:30px;
}

.result-box{
    padding:20px;
    border-radius:15px;
    background:#1e293b;
    box-shadow:0px 0px 15px rgba(0,0,0,0.3);
}

.prediction{
    font-size:26px;
    font-weight:bold;
    color:#38bdf8;
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DOWNLOAD MODEL
# =========================
MODEL_PATH = "my_image_classifier_model.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1GIM10m53KGe8pZHlPA7_gC9iMNgsJFjd"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# CLASS NAMES
# =========================
class_names = [
    "Covid",
    "Normal",
    "Viral Pneumonia"
]

# =========================
# HEADER
# =========================
st.markdown(
    "<div class='main-title'>🩻 Deteksi Penyakit Paru-Paru dari X-Ray</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>Klasifikasi Covid, Normal, dan Viral Pneumonia menggunakan Deep Learning</div>",
    unsafe_allow_html=True
)

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:
        st.image(
            image,
            caption="Gambar X-Ray",
            use_container_width=True
        )

    with col2:

        img = image.resize((224,224))

        img_array = np.array(img)
        img_array = img_array.astype(np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        probs = tf.nn.softmax(prediction[0]).numpy()

        predicted_class = np.argmax(probs)

        label = class_names[predicted_class]

        confidence = probs[predicted_class] * 100

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        st.markdown("## 📊 Hasil Analisis")

        if label == "Covid":
            st.error(f"🔴 Prediksi: {label}")

        elif label == "Normal":
            st.success(f"🟢 Prediksi: {label}")

        else:
            st.warning(f"🟡 Prediksi: {label}")

        st.markdown(
            f"<div class='prediction'>Confidence: {confidence:.2f}%</div>",
            unsafe_allow_html=True
        )

        st.progress(float(confidence/100))

        st.markdown("</div>", unsafe_allow_html=True)

        # =====================
        # TABLE
        # =====================
        st.markdown("### 📋 Probabilitas Semua Kelas")

        df = pd.DataFrame({
            "Kelas": class_names,
            "Probabilitas (%)": np.round(probs * 100, 2)
        })

        st.dataframe(
            df,
            use_container_width=True
        )

        # =====================
        # CHART
        # =====================
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

# =========================
# FOOTER
# =========================
st.markdown(
    """
    <div class='footer'>
    Sistem Deteksi Covid, Normal, dan Viral Pneumonia menggunakan CNN & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
