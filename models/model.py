import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

# Statics
data_dir = "../data/processed"
input_shape = (1024,1024,1)
latent_dim = 128

class CNNAutoencoderADModel(tf.keras.Model):

    def __init__(self):
        super(CNNAutoencoderADModel,self).__init__()
        
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

        # Decoder Part of the model:
        self.decoder = models.Sequential([
            layers.Dense(32 * 32 * 512, activation='relu', input_dim=latent_dim),  # Reshape to start decoding
            layers.Reshape((32, 32, 512)),
            layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='valid', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='valid', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='valid', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='valid', strides=2),
            layers.BatchNormalization(),
            layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='valid', strides=2),
            layers.BatchNormalization(),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Output layer with same size as input
        ])

    
    def call(self,x):
        """ inference """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded