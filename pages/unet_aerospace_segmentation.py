import streamlit as st
from PIL import Image
import torch
from io import BytesIO
import requests
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Encoder with reduced number of filters
        self.e11 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Decoder with corresponding reductions
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        # Use checkpointing for memory-intensive layers
        xe11 = F.relu(checkpoint(self.e11, x))
        xe12 = F.relu(checkpoint(self.e12, xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(checkpoint(self.e21, xp1))
        xe22 = F.relu(checkpoint(self.e22, xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(checkpoint(self.e31, xp2))
        xe32 = F.relu(checkpoint(self.e32, xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(checkpoint(self.e41, xp3))
        xe42 = F.relu(checkpoint(self.e42, xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(checkpoint(self.e51, xp4))
        xe52 = F.relu(checkpoint(self.e52, xe51))

        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(checkpoint(self.d11, xu11))
        xd12 = F.relu(checkpoint(self.d12, xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(checkpoint(self.d21, xu22))
        xd22 = F.relu(checkpoint(self.d22, xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(checkpoint(self.d31, xu33))
        xd32 = F.relu(checkpoint(self.d32, xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(checkpoint(self.d41, xu44))
        xd42 = F.relu(checkpoint(self.d42, xd41))

        out = self.outconv(xd42)

        return out

def load_model():
    model = UNet(n_class=1)  
    model.load_state_dict(torch.load('./models/model_weights.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess(image):
    image = image.resize((256, 256))
    image_np = np.array(image).astype(np.float32)
    if image_np.ndim == 2:  
        image_np = np.stack([image_np] * 3, axis=-1)
    
    image_tensor = torch.tensor(image_np).permute(2, 0, 1)  
    
    image_tensor = image_tensor.unsqueeze(0)  
    image_tensor /= 255.0  
    
    return image_tensor

model = load_model()

st.title('Forest Detection')

uploaded_files = st.file_uploader('Upload an image of a forest area', accept_multiple_files=True)
image_urls = st.text_area('Or enter image URLs (separate by commas)').split(',')

def predict(img):
    start_time = time.time()
    with torch.no_grad(): 
        output = model(img)
    
    probabilities = torch.sigmoid(output)

    binary_mask = (probabilities > 0.5).float()

    output_np = binary_mask.squeeze().cpu().numpy()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    return output_np, elapsed_time

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return img

def display_prediction(image, caption):
    processed_image = preprocess(image)  
    output_np, elapsed_time = predict(processed_image)

    st.image(image, caption=caption)
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
