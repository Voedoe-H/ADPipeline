import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', padding='valid', strides=2), # 1024x1024 -> (1024-3)/2 + 1 => 512x512x32
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
            layers.Input(shape=(latent_dim,)),
            layers.Dense(32 * 32 * 512, activation='relu'),  # Reshape to start decoding
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

def train_default_GPUK(learning_rate=1e-4, batch_size=4, epochs=5, validation_split=0.2, ssim_weight=0.5):

    tf.config.threading.set_intra_op_parallelism_threads(6)

    model = CNNAutoencoderADModel()
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}!")

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, [1024, 1024])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_size = len(image_paths)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    def combined_loss(y_true, y_pred):
        mse_loss = mse_loss_fn(y_true, y_pred)
        ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        ssim_loss = 1.0 - ssim_val  
        return (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss

    model.compile(optimizer=optimizer, loss=combined_loss)

    history = {'loss': [], 'val_loss': [], 'mse_loss' : [], 'val_mse_loss': [], 'ssim_loss': [], 'val_ssim_loss': []}
    for epoch in range(epochs):
       
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = 0
        train_mse_loss = 0
        train_ssim_loss = 0
        num_batches = 0
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                reconstructed = model(batch, training=True)
                mse_loss = mse_loss_fn(batch, reconstructed)
                ssim_val = tf.reduce_mean(tf.image.ssim(batch, reconstructed, max_val=1.0))
                ssim_loss = 1.0 - ssim_val
                loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss
            train_mse_loss += mse_loss
            train_ssim_loss += ssim_loss
            num_batches += 1

        train_loss /= num_batches
        train_mse_loss /= num_batches
        train_ssim_loss /= num_batches

        val_loss = 0
        val_mse_loss = 0
        val_ssim_loss = 0
        num_val_batches = 0
        for batch in val_dataset:
            reconstructed = model(batch, training=False)
            mse_loss = mse_loss_fn(batch, reconstructed)
            ssim_val = tf.reduce_mean(tf.image.ssim(batch, reconstructed, max_val=1.0))
            ssim_loss = 1.0 - ssim_val
            loss = (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss
            val_loss += loss
            val_mse_loss += mse_loss
            val_ssim_loss += ssim_loss
            num_val_batches += 1

        val_loss /= num_val_batches
        val_mse_loss /= num_val_batches
        val_ssim_loss /= num_val_batches

        history['loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['mse_loss'].append(float(train_mse_loss))
        history['val_mse_loss'].append(float(val_mse_loss))
        history['ssim_loss'].append(float(train_ssim_loss))
        history['val_ssim_loss'].append(float(val_ssim_loss))
        print(f"Train Loss: {train_loss:.4f} (MSE: {train_mse_loss:.4f}, SSIM: {train_ssim_loss:.4f}) - "
              f"Val Loss: {val_loss:.4f} (MSE: {val_mse_loss:.4f}, SSIM: {val_ssim_loss:.4f})")

    model_save_path = "trained_model"
    dummy_input = tf.random.normal([1, 1024, 1024, 1])
    _ = model(dummy_input)
    model.save(model_save_path)
    return model, history


def train_default_GPU(learning_rate=1e-4, batch_size=32, epochs=5, validation_split=0.2, ssim_weight=0.5):
    # Optimize threading for GPU training and multitasking
    tf.config.threading.set_intra_op_parallelism_threads(2)  # For data pipeline/I/O
    tf.config.threading.set_inter_op_parallelism_threads(2)  # For parallel ops
    tf.debugging.set_log_device_placement(True)  # Log GPU usage (remove after testing)

    # Initialize model
    model = CNNAutoencoderADModel()

    # Load image paths
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}!")

    # Image preprocessing
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)
        image.set_shape([None, None, 1])
        image = tf.image.resize(image, [1024, 1024])
        image = tf.cast(image, tf.float32) / 255.0
        return image, image  # Autoencoder: input = target

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_size = len(image_paths)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset = dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Define loss function
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    def combined_loss(y_true, y_pred):
        mse_loss = mse_loss_fn(y_true, y_pred)
        ssim_val = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        ssim_loss = 1.0 - ssim_val
        return (1 - ssim_weight) * mse_loss + ssim_weight * ssim_loss

    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[mse_loss_fn, 'mae'])

    # Train with model.fit for GPU efficiency
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(
                    f"SSIM Loss: {1.0 - tf.reduce_mean(tf.image.ssim(model.predict(train_dataset.take(1))[0], next(iter(train_dataset))[0], max_val=1.0)):.4f}, "
                    f"Val SSIM Loss: {1.0 - tf.reduce_mean(tf.image.ssim(model.predict(val_dataset.take(1))[0], next(iter(val_dataset))[0], max_val=1.0)):.4f}"
                )
            )
        ]
    )

    # Save model
    model_save_path = "trained_model"
    model.save(model_save_path)

    return model, history


train_default_GPU()