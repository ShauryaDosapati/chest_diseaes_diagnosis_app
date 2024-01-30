from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import streamlit as st
st.title("Shaurya - Chest X-Ray Diagnosis App")
model=load_model('chest_xray.h5')

training_classes_indices={'COVID-19': 0, 'LUNG-CANCER': 1, 'NORMAL': 2, 'PNEUMONIA': 3}

uploaded_file = st.file_uploader("Choose an x-ray image...",type=['png','jpg','jpeg'])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(img,caption='Uploaded Chest X-Ray Image..', use_column_width=True)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    result = int(classes[0][0])

    indices_class = {v: k for k, v in training_classes_indices.items()}

    # Getting the index of the class with the highest probability
    predicted_class_index = np.argmax(classes, axis=1)

    # Mapping the index to the class name
    predicted_class = [indices_class[i] for i in predicted_class_index]
    print("Predicted Class:", predicted_class)
    st.title("Predicted Label:" + str(predicted_class))
