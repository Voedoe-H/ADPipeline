import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statics
data_dir = os.path.abspath("../data/processed")
input_shape = (1024,1024,1)
latent_dim = 128

class CNNAutoencoderADModel(tf.keras.Model):

    def __init__(self):
        super(CNNAutoencoderADModel,self).__init__()
        
        # Down Sampling Rate Computation
        # O := Output Size
        # I := Input Size
        # K := Kernel Size
        # S := Stride 
        # O = (I-K)/S + 1

        # Encoder Part of the model:
        self.encoder = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='valid', strides=2, input_shape=input_shape), # 1024x1024 -> (1024-3)/2 + 1 => 512x512x32
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='valid', strides=2), # 512x512x3 -> (512-3)/2 + 1 => 255x255x64
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='valid', strides=2), # 255x255x64 -> (255-3)/2 + 1 => 127x127x128
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='valid', strides=2), # 127x127x128 -> (127-3)/2 + 1 => 63x63x256
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='valid', strides=2), # 63x63x256 -> (63-3)/2 + 1 => 31x31x512
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(latent_dim)
        ])

        # Up Sampling Rate Computaiton
        # O := 
        #
        #

        # Decoder Part of the model:
        self.decoder = models.Sequential([
            layers.Dense(32 * 32 * 512, activation='relu', input_dim=latent_dim),  # Reshape to start decoding
            layers.Reshape((32, 32, 512)),
            layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same', strides=2),
            layers.BatchNormalization(),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Output layer with same size as input
        ])

    
    def call(self,x):
        """ inference """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def test_output_shape():
    """ Simple function to test if the encoding and decoding provide the desired dimensions """
    model = CNNAutoencoderADModel()
    sample_input = np.random.rand(1, 1024, 1024, 1).astype(np.float32)
    encoded_output = model.encoder(sample_input)
    decoded_output = model.decoder(encoded_output)

    print(f"Encoded shape: {encoded_output.shape}")
    print(f"Decoded shape: {decoded_output.shape}")

def ssim_metric(y_true, y_pred):
    """ Function to track the structural similarity index during training """
    # Results are between -1 and 1 the higher the better
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def psnr_metric(y_true, y_pred):
    """ Function to track the precision recall curve during training """
    # Results are in db -> Higher values mean better reconstruction
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def train_model():
    """ Funciton to train/retrain the model based on the data provided in the data directory """
    # Create Model
    model = CNNAutoencoderADModel()

    # Load Training Data
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}!")

    df = pd.DataFrame({"filename": image_paths})

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="filename",
        target_size=(1024, 1024),
        color_mode="grayscale",
        batch_size=32,
        class_mode=None,
        shuffle=True
    )
    
    for i in range(5):
        batch = next(train_generator)  # this will get the next batch
        # Check if the batch is correct
        print(f"Batch {i}: min = {batch[0].min()}, max = {batch[0].max()}")

    batch = next(iter(train_generator))  # Get a sample batch
    print(f"Batch shape: {batch.shape}")  # Should be (batch_size, 1024, 1024, 1)
    steps_per_epoch = len(train_generator)
    print(f"Steps per epoch: {steps_per_epoch}") 
    # Define the training setup
    model.compile(optimizer='adam', 
              loss='mse',  
              metrics=['mae', psnr_metric, ssim_metric])
    print("Batch contains None:", np.any(batch == None))  # Should be False
    print("Batch min:", np.min(batch), "Batch max:", np.max(batch)) 
    model.build(input_shape=(None, 1024, 1024, 1))
    model.summary()
    # Fit the model
    model.fit(train_generator, epochs=10,steps_per_epoch=len(train_generator))

def train_default():
    model = CNNAutoencoderADModel()
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}!")
    print(f"Number Pictures:{len(image_paths)}")    

train_default()