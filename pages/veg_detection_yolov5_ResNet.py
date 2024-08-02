import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
from torchvision.io import read_image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import requests
# from model_utils_k import ResNet18ForLocalization,load_image_from_url, predict_and_display, load_history, plot_history, load_yolov5_model, save_image_to_bytes

import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import requests
from torchvision.models import resnet18, ResNet18_Weights
import streamlit as st
import json

class ResNet18ForLocalization(nn.Module):
    def __init__(self):
        super(ResNet18ForLocalization, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
       
        self.fc1 = nn.Linear(1000, 1024)
        self.fc2 = nn.Linear(1024, 4)  # 4 для bbox
        self.fc3 = nn.Linear(1024, 3)  # 3 класса
        self.dropout = nn.Dropout() 
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.fc3.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.model(x)
        x = self.fc1(features)
        x = torch.relu(x)
        bbox_outputs = self.fc2(x)
        class_outputs = self.fc3(x)
        return bbox_outputs, class_outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
model = ResNet18ForLocalization().to(device)
model.load_state_dict(torch.load('models/resnet_weight.pt', map_location=torch.device('cpu')))
ind2class = {0: 'cucumber', 1: 'eggplant', 2: 'mushroom'}
    
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def predict_and_display(image):
    # Преобразования
    transform = T.Compose([T.Resize((227, 227)), T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        bbox_pred, class_pred = model(image_tensor)
    
    # Преобразуйте предсказания
    bbox_pred = bbox_pred.cpu().numpy().flatten()
    class_pred = class_pred.cpu().numpy().flatten()
    
    xmin, ymin, xmax, ymax = bbox_pred
    width, height = image.size
    xmin = int(xmin * width)
    ymin = int(ymin * height)
    xmax = int(xmax * width)
    ymax = int(ymax * height)
    
    class_index = class_pred.argmax()
    class_label = ind2class[class_index]
    
    # Отобразите изображение с рамкой
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    plt.text(xmin, ymin, class_label, color='red', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='red', boxstyle='round,pad=0.5'))
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf, caption='Prediction Result', use_column_width=True)
    plt.close()
    
@st.cache_data  
def load_history(file_path):
    with open(file_path, 'r') as f:
        logs = json.load(f)
    return logs

@st.cache_data
def plot_history(history, grid=True):
    fig, ax = plt.subplots(2, 2, figsize=(18, 10))
    
    # Losses
    ax[0, 0].plot(history['train_losses'], label='train loss')
    ax[0, 0].plot(history['valid_losses'], label='valid loss')
    ax[0, 0].set_title(f'Loss on epoch {len(history["train_losses"])}')
    ax[0, 0].grid(grid)
    ax[0, 0].legend()
    
    ax[0, 1].plot(history['train_losses_bbox'], label='train bbox loss')
    ax[0, 1].plot(history['valid_losses_bbox'], label='valid bbox loss')
    ax[0, 1].set_title(f'Bounding Box Loss on epoch {len(history["train_losses"])}')
    ax[0, 1].grid(grid)
    ax[0, 1].legend()
    
    # Accuracy
    ax[1, 0].plot(history['train_accs'], label='train acc')
    ax[1, 0].plot(history['valid_accs'], label='valid acc')
    ax[1, 0].set_title(f'Accuracy on epoch {len(history["train_losses"])}')
    ax[1, 0].grid(grid)
    ax[1, 0].legend()
    
    # F1 Scores
    ax[1, 1].plot(history['train_f1s'], label='train F1-score')
    ax[1, 1].plot(history['valid_f1s'], label='valid F1-score')
    ax[1, 1].set_title(f'F1 Score on epoch {len(history["train_losses"])}')
    ax[1, 1].grid(grid)
    ax[1, 1].legend()
    
    # Customizing subplot appearance
    for a in ax.flat:
        a.set_xlabel('Epoch')
        a.set_ylabel('Value')

    plt.tight_layout()
    buf = BytesIO()
    try:
        plt.savefig(buf, format='png')
        buf.seek(0)
    except Exception as e:
        print(f"Error saving figure: {e}")
        buf = None
    finally:
        plt.close()
    
    return buf
        
        
@st.cache_data
def load_yolov5_model(model_path):
    # Загрузка кастомной модели YOLOv5
    model = torch.hub.load(
        repo_or_dir='ultralytics/yolov5',
        model='custom',
        path=model_path
    )
    model.conf = 0.1  # Установка порога уверенности
    return model


def save_image_to_bytes(image):
    buf = BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return buf


    
    
st.header('Object Localization in Veg Dataset (ResNet+YOLO)')

st.title('Локализация огурчиков')

model_option = st.selectbox('Выберите модель', ['ResNet18', 'YOLOv5'])

if model_option == 'ResNet18':
    
    st.subheader('Dataset Information')
    history = load_history('pages/training_history.json')
    buf = plot_history(history)
    st. image(buf)
    
    dataset_info = {
        'Number of classes': 3,
        'Number of training images': 150,
        'Number of validation images': 30,
        'Training time, sec': 52 
    }
    st.write(dataset_info)
    option = st.selectbox('Выберите способ загрузки', ['From URL', 'From File'])

    if option == 'From URL':
        url = st.text_input('Image URL', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWgudRC2zhtWViDI0JkDt9fS1GTPQ-exJYEg&s')
        if st.button('Classify'):
            if url:
                image = load_image_from_url(url)
                predict_and_display(image)
            else:
                st.error("Please enter a valid URL.")

    elif option == 'From File':
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                predict_and_display(image)
                

if model_option == 'YOLOv5':
    
    img1 = Image.open('images/confusion_matrix_K.png')
    img2 = Image.open('images/PR_curve_K.png')
    img3 = Image.open('images/results_K.png')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img1, caption='Confusion Matrix', use_column_width=True)
    with col2:
        st.image(img2, caption='PR Curve', use_column_width=True)
    with col3:
        st.image(img3, caption='Metrics', use_column_width=True)
        
    dataset_info = {
        'Number of classes': 3,
        'Number of training images': 393,
        'Number of validation images': 36,
        'Training time, m': 15 
    }
    st.write(dataset_info)
    
    model_path = 'models/best_k.pt'
    model = load_yolov5_model(model_path)

    option = st.selectbox('Выбери способ загрузки изображения', ['From URL', 'From File'])

    if option == 'From URL':
        url = st.text_input('Image URL', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWgudRC2zhtWViDI0JkDt9fS1GTPQ-exJYEg&s')
        if st.button('Classify'):
            if url:
                image = load_image_from_url(url)
                model.eval()
                with torch.no_grad():
                    results = model(image)
                
                results_img = results.render()[0]
                buf = save_image_to_bytes(Image.fromarray(results_img))
                
                st.image(buf, caption='YOLOv5 Prediction')
            else:
                st.error("Please enter a valid URL.")

    elif option == 'From File':
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                model.eval()
                with torch.no_grad():
                    results = model(image)
                
                # Convert results to PIL Image
                results_img = results.render()[0]  # Get the first image with results
                buf = save_image_to_bytes(Image.fromarray(results_img))
                
                st.image(buf, caption='YOLOv5 Prediction')
    