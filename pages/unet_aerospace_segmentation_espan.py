import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import requests
import time
from torchvision import transforms as T
import segmentation_models_pytorch as smp
import numpy as np

import streamlit as st

st.header('Semantic Segmentation with U-Net')

st.header('Model Training Information')
num_epochs = 15

st.write(f"Number of Epochs: {num_epochs}")
st.write(f"Size of Train Dataset: {3500}, Validation: {600}, Test: {1000} images")
st.write("Test loss: 0.397679 | Test IoU: 0.730058 | Test accuracy: 0.8155")

acc_image_path = './images/tolu_acc.png'
iou_image_path = './images/tolu_iou.png'
loss_image_path = './images/tolu_loss.png'

st.header('Training Performance Plots')

st.image(acc_image_path, caption='Accuracy per Epoch', use_column_width=True)
st.image(iou_image_path, caption='IoU per Epoch', use_column_width=True)
st.image(loss_image_path, caption='Loss per Epoch', use_column_width=True)

model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights="imagenet", 
                 classes=1, activation=None, encoder_depth=5, decoder_channels=[512, 256, 64, 32, 16])

model_weights_path = './models/unet.pt'  
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
model.eval()

st.title('Forest Detection')

uploaded_files = st.file_uploader('Upload an image of a forest area', accept_multiple_files=True)
image_urls = st.text_area('Or enter image URLs (separate by commas)').split(',')

def preprocess_image(image):
    preprocess = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return preprocess(image).unsqueeze(0)  

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

def predict(img):
    start_time = time.time()

    img_tensor = preprocess_image(img).to(DEVICE)  

    with torch.no_grad():
        output = model(img_tensor)
    
    probabilities = torch.sigmoid(output)
    binary_mask = (probabilities).float()

    output_np = binary_mask.squeeze().cpu().numpy()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output_np, elapsed_time

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def display_prediction(image, caption):
    output_np, elapsed_time = predict(image)
    st.image(image, caption=caption, use_column_width=True)
    st.image(output_np, caption="Segmented Mask", use_column_width=True, clamp=True)
    st.write(f'Model Response Time: {elapsed_time:.4f} seconds')

if uploaded_files:
    for file in uploaded_files:
        pil_image = Image.open(file).convert('RGB') 
        display_prediction(pil_image, caption=f'Uploaded Image: {file.name}')

if image_urls:
    for url in image_urls:
        url = url.strip()
        if url:
            try:
                pil_image = load_image_from_url(url)
                display_prediction(pil_image, caption=f'Image from URL: {url}')
            except Exception as e:
                st.error(f"Error loading image from URL {url}: {e}")
