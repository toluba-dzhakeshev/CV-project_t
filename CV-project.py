import streamlit as st

# Заголовок страницы
st.title('Computer Vision Project Dashboard')

# Введение
st.header('Welcome to the CV Project Dashboard!')
st.write(
    "This multipage application showcases various computer vision models and their applications. "
    "Here, you'll find tools for object detection and semantic segmentation, each tailored for specific tasks."
)

# Обзор проекта
st.header('Project Overview')

st.write(
    "In this project, we explore and implement different computer vision techniques. The main pages of the application include:"
)

# Список страниц
st.subheader('1. Object Localization in Veg Dataset (ResNet+YOLO)')
st.write(
    "This page demonstrates object localization using a model trained on the Veg dataset. "
    "Check out the performance and results of the model with visual examples."
)

st.subheader('2. Object Detection with YOLO')
st.write(
    "Here, we apply the YOLO (You Only Look Once) model to detect ships in aerial images. "
    "This page supports file uploads directly or via URL for immediate detection."
)

st.subheader('3. Semantic Segmentation with U-Net')
st.write(
    "This section utilizes the U-Net model for semantic segmentation of aerospace images. "
    "Explore the segmentation results and see how the model identifies different regions in the images."
)