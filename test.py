import tensorflow as tf

# List physical devices
devices = tf.config.list_physical_devices()
print("Available devices:", devices)

# Specifically check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
else:
    print("No GPU detected.")