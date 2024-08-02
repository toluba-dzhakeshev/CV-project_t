import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO

st.header('Object Detection with YOLO')

st.header('Model Training Information')

# Информация о процессе обучения
num_epochs = 50
train_size = 13400
validation_size = 1340
test_size = 2680

st.write(f"Number of Epochs: {num_epochs}")
st.write(f"YOLOv5 Type: M")
st.write(f"Size of Train Dataset: {train_size}, Validation: {validation_size}, Test: {test_size} images")

# Метрики (примерные, добавьте свои графики и матрицы)
results = './images/nat_results.png'
PR_curve = './images/nat_PR_curve.png'

st.header('Training Performance Plots')
st.image(results, caption='Results', use_column_width=True)
st.image(PR_curve, caption='PR Curve', use_column_width=True)

@st.cache_resource
def get_model(conf):
    model = torch.hub.load(
        repo_or_dir='ultralytics/yolov5',
        model='custom',
        path='./models/yolov5/best_n.pt'
    )
    model.eval()
    model.conf = conf
    print('Model loaded')
    return model

def load_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def predict_and_display(model, image):
    results = model(image)
    return results

# Слайдер для конфиденциальности модели
t = st.slider('Model Confidence Threshold', 0.0, 1.0, 0.1)

with st.spinner('Loading model...'):
    model = get_model(t)

# UI элементы
st.write("Choose the method to upload images:")

option = st.selectbox('Select upload method', ['From URL', 'From File'])

if option == 'From URL':
    url = st.text_input('Image URL', 'https://example.com/image.jpg')
    if st.button('Classify'):
        if url:
            image = load_image_from_url(url)
            with st.spinner('Processing image...'):
                results = predict_and_display(model, image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.image(results.render(), caption='Prediction Results', use_column_width=True)
        else:
            st.error("Please enter a valid URL.")

elif option == 'From File':
    uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            with st.spinner('Processing image...'):
                results = predict_and_display(model, image)
            lcol, rcol = st.columns(2)  # Создание двух столбцов для отображения
            with lcol:
                st.image(image, caption='Uploaded Image', use_column_width=True)
            with rcol:
                st.image(results.render(), caption='Prediction Results', use_column_width=True)