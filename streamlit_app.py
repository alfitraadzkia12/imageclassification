import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("Deteksi Covid dari X-Ray")

MODEL_PATH = "my_image_classifier_model.h5"
FILE_ID = "1AABBF6Zh23ONuTxfw2baOPjmnubGZbhu"

if not os.path.exists(MODEL_PATH):
gdown.download(
f"https://drive.google.com/uc?id={FILE_ID}",
MODEL_PATH,
quiet=False
)

@st.cache_resource
def load_my_model():
return tf.keras.models.load_model(MODEL_PATH)

model = load_my_model()

class_names = [
"Covid",
"Normal",
"Viral Pneumonia"
]

uploaded_file = st.file_uploader(
"Upload Gambar X-Ray",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

```
image = Image.open(uploaded_file).convert("RGB")

st.image(
    image,
    caption="Gambar yang diupload",
    use_container_width=True
)

if st.button("Prediksi"):

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    pred_idx = np.argmax(prediction)

    confidence = float(
        np.max(prediction)
    ) * 100

    st.success(
        f"Hasil: {class_names[pred_idx]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )
```

