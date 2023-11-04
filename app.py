# importing all the packages
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torch
from skorch import NeuralNetClassifier
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Cnn





# Reading the data
mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
X /= 255.0

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
XCnn = X.reshape(-1, 1, 28, 28) #reshape input 
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42) #train test split

torch.manual_seed(0)



# reshape and train test split

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
XCnn = X.reshape(-1, 1, 28, 28)
XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.25, random_state=42)

from PIL import Image
import torchvision.transforms as transforms

torch.manual_seed(0)


model=Cnn()

# Specify the path to the saved model weights
model_weights_path = 'model_weights.pth'

# Load the model weights
model.load_state_dict(torch.load(model_weights_path))

# Set the model to evaluation mode for inference
model.eval()

# Create a NeuralNetClassifier using the loaded model
cnn = NeuralNetClassifier(
    module=model,
    max_epochs=0,  # Set max_epochs to 0 to avoid additional training
    lr=0.002,  # You can set this to the learning rate used during training
    optimizer=torch.optim.Adam,  # You can set the optimizer used during training
    device='cpu'  # You can specify the device ('cpu' for CPU, 'cuda' for GPU, etc.)
)

cnn.fit(XCnn_train, y_train)



# Set the page title
st.title("Handwritten Text Digit Recognition")



stroke_width = st.sidebar.slider("Stroke width: ", 1, 35, 32)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform", "polygon")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)
#create canvas component
canvas_result = st_canvas(
    fill_color="white",  # Set the canvas background color to white
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    update_streamlit=realtime_update,
    height=28,  # Set the canvas height to 28 pixels
    width=28,  # Set the canvas width to 28 pixels
    drawing_mode=drawing_mode,
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="full_app",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    image = canvas_result.image_data
    image1 = image.copy()
    image1 = image1.astype('uint8')
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image1 = cv2.resize(image1, (28, 28))
    st.image(image1)

    # Correctly reshape the image
    image1 = image1.reshape(1, 1, 28, 28).astype('float32')
    st.title(np.argmax(model.predict(image1)))

if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))

