import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from PIL import Image


# Focusing on the MVTec AD data set, specifically the screws

# Statics
original_data_path = "./raw/screw/train/good" # Path to the screw images that were pre classfied as good/ not an anomaly
output_data_path = "./processed" # Path to the dir where the post processing images are saved at

def image_analysis():
    """ Simple manual analysis function to see what pictures im actually working with in this data set """
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

    image_counter = 0

    for path in image_paths:
        img = Image.open(path)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        i = 0
        for batch in imagegen.flow(img_array, batch_size=1, save_to_dir=output_data_path, save_prefix='aug', save_format='jpeg'):
            i += 1
            if i > 3:  # Generate 3 augmented images per original image (you can adjust this)
                break
        image_counter += 1
    
    augmented_images = len([f for f in os.listdir(output_data_path) if f.startswith('aug')])
    print(f"Total augmented images saved: {augmented_images}")

#good_image_pre_processing() 
