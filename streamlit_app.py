import streamlit as st
from PIL import Image

st.title("Deteksi Covid dari X-Ray")

uploaded_file = st.file_uploader(
    "Upload Gambar X-Ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang Diupload",
        use_container_width=True
    )

    st.success(
        "Model berhasil dimuat (Demo Streamlit)"
    )
