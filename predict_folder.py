import tensorflow as tf
import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing import image

# Load model
from tensorflow.keras.models import load_model  # Ensure you have this

model_path = os.path.join(os.path.dirname(__file__), 'temple_classifier_model.h5')
model = load_model(model_path)
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model_path = resource_path("temple_classifier_model.h5")

# Example with Keras
from tensorflow.keras.models import load_model
model = load_model(model_path)


# Map class indices to labels
class_indices = {
    0: 'Lakshmi_Narasimha',
    1: 'Main_Balaram_Lotus_Balaram_Face',
    2: 'Main_Balaram_Lotus_Balaram_Feet',
    3: 'Main_Krishna_Balaram_Full',
    4: 'Main_Krishna_Lotus_Face',
    5: 'Main_Krishna_Lotus_Feet',
    6: 'Main_Srila_Prabhupada',
    7: 'Mangalarati_Darshan',
    8: 'Nitai_Gauranga',
    9: 'Tulasi_Maharani',
    10: 'Utsava_Krishna_Balaram',
    11: 'Utsava_Srila_Prabhupada'
}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)

    return class_indices[class_idx]

# Folder of unsorted images
import tkinter as tk
from tkinter import filedialog

# Hide root window
root = tk.Tk()
root.withdraw()

# Prompt user to select folders
input_folder = filedialog.askdirectory(title="Select folder with unsorted images")
if not input_folder:
    print("❌ No input folder selected. Exiting.")
    exit()

output_base = filedialog.askdirectory(title="Select folder to save sorted images")
if not output_base:
    print("❌ No output folder selected. Exiting.")
    exit()


# Create output folders if they don’t exist
for label in class_indices.values():
    os.makedirs(os.path.join(output_base, label), exist_ok=True)

# Process each image
for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, img_name)
        predicted_class = predict_image(img_path)
        dest_folder = os.path.join(output_base, predicted_class)
        shutil.move(img_path, os.path.join(dest_folder, img_name))
        print(f"{img_name} ➜ {predicted_class}")
