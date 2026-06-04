
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

model = load_model('my_image_classifier_model.h5')

nama_file = input("Masukkan nama file: ")

folder_dataset = "/content/dataset/Covid19-dataset/test"

img_path = None

for folder in ["Covid", "Normal", "Viral Pneumonia"]:
    path = os.path.join(folder_dataset, folder, nama_file)

    if os.path.exists(path):
        img_path = path
        break

if img_path is None:
    print("File tidak ditemukan!")

else:
    img = load_img(img_path, target_size=(224,224))

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_names = ['Covid', 'Normal', 'Viral Pneumonia']

    score = np.argmax(prediction)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("Nama File :", nama_file)
    print("Prediksi Model :", class_names[score])
