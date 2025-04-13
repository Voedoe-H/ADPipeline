import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
import platform
import sys
import tf2onnx

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

def train_default(learning_rate=1e-4, batch_size=32, epochs=10, validation_split=0.2):
    model = CNNAutoencoderADModel()
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}!")

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=1, expand_animations=False)  # Set channels=1 for grayscale
        image.set_shape([None, None, 1])  # Explicit shape: height x width x 1
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
    # Define optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn)
    
    # Training history
    history = {
        'loss': [],
        'val_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Training
        train_loss = 0
        num_batches = 0
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                reconstructed = model(batch, training=True)
                loss = loss_fn(batch, reconstructed)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss += loss
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        val_loss = 0
        num_val_batches = 0
        for batch in val_dataset:
            reconstructed = model(batch, training=False)
            loss = loss_fn(batch, reconstructed)
            val_loss += loss
            num_val_batches += 1
        
        val_loss /= num_val_batches
        
        # Store history
        history['loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return model, history

stop_training = False

def listen_for_stop_key():
    global stop_training
    if platform.system() == "Windows":
        import msvcrt
        while not stop_training:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key.lower() == b's':
                    stop_training = True
                    print("\nStopping training early...")
                    break
            time.sleep(0.1)
    else:
        import select
        while not stop_training:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 's':
                    stop_training = True
                    print("\nStopping training early...")
                    break

def train_default_GPU(learning_rate=1e-4, batch_size=4, epochs=4, validation_split=0.2):
    global stop_training
    stop_training = False
    threading.Thread(target=listen_for_stop_key, daemon=True).start()
    print("Press 's' to stop training early.")

    tf.config.threading.set_intra_op_parallelism_threads(2)

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
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer=optimizer, loss=loss_fn)

    history = {'loss': [], 'val_loss': []}
    for epoch in range(epochs):
        if stop_training:
            print("Training stopped early.")
            break

        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = 0
        num_batches = 0
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                reconstructed = model(batch, training=True)
                loss = loss_fn(batch, reconstructed)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss += loss
            num_batches += 1
                
        train_loss /= num_batches

        val_loss = 0
        num_val_batches = 0
        for batch in val_dataset:
            reconstructed = model(batch, training=False)
            loss = loss_fn(batch, reconstructed)
            val_loss += loss
            num_val_batches += 1

        val_loss /= num_val_batches

        history['loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        print(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    model_save_path = "trained_model" 
    dummy_input = tf.random.normal([1, 1024, 1024, 1])
    _ = model(dummy_input)  # Triggers model building
    model.save(model_save_path)
    #model.save(model_save_path,save_format="tf")
    return model, history

train_default_GPU()