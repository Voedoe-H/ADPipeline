import tf2onnx
import tensorflow as tf
import onnxruntime as ort
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2

def convert():
    # Path to SavedModel dir
    saved_model_dir = "trained_model"
    onnx_model_path = "model.onnx"

    # Load the model
    model = tf.keras.models.load_model(saved_model_dir)

    # Dummy input shape (batch size 1, 1024x1024, grayscale)
    spec = (tf.TensorSpec((1, 1024, 1024, 1), tf.float32, name="input"),)

    # Convert the model
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    # Save the ONNX model
    with open(onnx_model_path, "wb") as f:
        f.write(model_proto.SerializeToString())

    print("ONNX model saved to:", onnx_model_path)

def highlight_large_errors(original, reconstructed, threshold=0.05):
    original = original.astype(np.float32) / 255.0
    reconstructed = reconstructed.astype(np.float32) / 255.0

    if original.shape[-1] == 1:
        original = original.squeeze(-1)
    if reconstructed.shape[-1] == 1:
        reconstructed = reconstructed.squeeze(-1)

    diff = np.abs(original - reconstructed)  # Shape: (H, W)

    mask = diff > threshold  # Shape: (H, W)

    original_rgb = np.stack([original] * 3, axis=-1)  # Shape: (H, W, 3)

    highlight = original_rgb.copy()
    highlight[mask] = [1.0, 0.0, 0.0]  # Red

    return highlight, diff



def plot_highlighted_defect(original, reconstructed, diff, highlight, title="Detected Defect", threshold=0.005):
    original = original.squeeze()
    reconstructed = reconstructed.squeeze()
    diff = diff.squeeze()
    highlight = highlight.squeeze()

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(reconstructed, cmap='gray')
    axs[1].set_title("Reconstructed")
    axs[1].axis('off')

    axs[2].imshow(diff, cmap='hot')
    axs[2].set_title("Abs Difference")
    axs[2].axis('off')

    mask = diff > threshold
    highlight_overlay = np.stack([original] * 3, axis=-1)  
    highlight_overlay[mask] = [1.0, 0.0, 0.0] 

    axs[3].imshow(highlight_overlay)
    axs[3].set_title("Highlighted Defect")
    axs[3].axis('off')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def img_pre(path):
    img = load_img(path,color_mode='grayscale',target_size=(1024,1024))
    img_arr = img_to_array(img) / 255.0
    return np.expand_dims(img_arr,axis=0)


def heatmap_inference():
    model = tf.keras.models.load_model("trained_model")


def test_inference():
    
    model = tf.keras.models.load_model("trained_model")
    
    test_dirs = {
        "manipulated_front_dir" : os.path.abspath("../data/screw/test/manipulated_front"),
        "scratch_head_dir" : os.path.abspath("../data/screw/test/scratch_head"),
        "scratch_neck_dir" : os.path.abspath("../data/screw/test/scratch_neck"),
        "thread_side_dir" : os.path.abspath("../data/screw/test/thread_side"),
        "thread_top_dir" : os.path.abspath("../data/screw/test/thread_top")
    }

    overall_losses = []
   
    for name, dir_path in test_dirs.items():
        print(dir_path)
        image_paths = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if f.endswith((".png",".jpg",".jpeg"))]
        losses = []
        for path in image_paths:
            print(path)
            x = img_pre(path)
            x_reconstructed = model.predict(x, verbose=0)
            mse = mean_squared_error(x.flatten(), x_reconstructed.flatten())
            losses.append(mse)
            x = x.squeeze()  # (1024, 1024)
            x_reconstructed = x_reconstructed.squeeze()  # (1024, 1024)
            diff = np.abs(x - x_reconstructed)
            plot_highlighted_defect(x, x_reconstructed, diff, highlight=x_reconstructed)
        #plt.figure(figsize=(10, 4))
        #plt.plot(losses, marker='o')
        #plt.title(f"Reconstruction Losses - {os.path.basename(dir_path)}")
        #plt.xlabel("Image Index")
        #plt.ylabel("MSE Loss")
        #plt.grid(True)
        #plt.tight_layout()
        #plt.show()

    good_losses = []

    good_dir = os.path.abspath("../data/screw/test/good")
    image_paths = [os.path.join(good_dir,f) for f in os.listdir(good_dir) if f.endswith((".png"))]
    for path in image_paths:
        x = img_pre(path)
        x_reconstructed = model.predict(x, verbose=0)
        mse = mean_squared_error(x.flatten(), x_reconstructed.flatten())
        good_losses.append(mse)
    plt.figure(figsize=(10, 4))
    plt.plot(good_losses, marker='o')
    plt.title(f"Reconstruction Losses Good")
    plt.xlabel("Image Index")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # 0.005 boundary hypothesi
#convert()

#session = ort.InferenceSession("model.onnx")

# Create dummy input
#dummy_input = np.random.rand(1, 1024, 1024, 1).astype(np.float32)

# Run inference
#input_name = session.get_inputs()[0].name
#outputs = session.run(None, {input_name: dummy_input})

#print("Inference output shape:", outputs[0].shape)

test_inference()