import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from PIL import Image


# Focusing on the MVTec AD data set, specifically the screws

# Statics
original_data_path = "./raw/screw/train/good"
output_data_path = "./processed"

def image_analysis():
    img_path = "./raw/screw/train/good/000.png"
    img = Image.open(img_path)
    width, height = img.size
    print(f"Dimension: {width} x {height}")
    print(f"Image mode: {img.mode}")
    image_array = np.array(img)
    print(f"Array shape: {image_array.shape}")
    pixel_min_value = np.min(image_array)
    pixel_max_value = np.max(image_array)
    print(f"Pixel value range: {pixel_min_value}, {pixel_max_value}")
    print(f"Data type: {image_array.dtype}")


def good_image_pre_processing():
    """ Processing function that takes the original 319 images that represent the non anomaly case and does certain preprocessings to achieve a more generalizeed total data set of 1k images """
    # Just for learning purposes the augmentation process is not done directly on the batches during training but pre training
    
    image_paths = [os.path.join(original_data_path, f) for f in os.listdir(original_data_path) if f.endswith('.png') or f.endswith('.jpg')]
    
    imagegen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode='nearest',
        rescale = 1./255
    )

    for path in image_paths:
        pass
    
good_image_pre_processing() 
