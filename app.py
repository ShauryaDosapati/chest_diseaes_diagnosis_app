from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import streamlit as st

# Load your model
model = load_model('chest_xray.h5')
training_classes_indices = {'COVID-19': 0, 'LUNG-CANCER': 1, 'NORMAL': 2, 'PNEUMONIA': 3}

# Define your images (place them in the same folder as your script)
thumbnail_images = {
    'Covid-19 Sample': 'samples/Covid19-sample.jpg',
    'Lung Cancer CT Sample': 'samples/lung-cancer-ct-scan-sample1.jpg',
    'PNEUMONIA Sample': 'samples/PNEUMONIA_sample1.jpeg',
    'Normal Sample': 'samples/Normal-sample1.jpeg'
}

st.title("Shaurya - Chest X-Ray Diagnosis App")

# Sidebar - Thumbnail selection
st.sidebar.title('Sample X-Rays')
selected_thumbnail = st.sidebar.radio("Choose a sample x-ray:", list(thumbnail_images.keys()))

# Display the selected thumbnail
st.sidebar.image(thumbnail_images[selected_thumbnail], width=150, caption=selected_thumbnail)

# Handle file upload or use the selected thumbnail
uploaded_file = st.file_uploader("Choose an x-ray image...", type=['png', 'jpg', 'jpeg'])
if not uploaded_file:
    uploaded_file = thumbnail_images[selected_thumbnail]

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Chest X-Ray Image..', use_column_width=True)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = model.predict(img_data)
    result = int(classes[0][0])

    indices_class = {v: k for k, v in training_classes_indices.items()}
    predicted_class_index = np.argmax(classes, axis=1)
    predicted_class = [indices_class[i] for i in predicted_class_index]

    st.title("Predicted Label:" + str(predicted_class))
