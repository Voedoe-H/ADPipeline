import tf2onnx
import tensorflow as tf
import onnxruntime as ort
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import cv2
import torch
from pytorch_msssim import ssim

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

def plot_histograms(good_errors, bad_errors):
    plt.figure(figsize=(10, 6))
    plt.hist(good_errors, bins=50, alpha=0.6, label='Good', color='green')
    plt.hist(bad_errors, bins=50, alpha=0.6, label='Bad', color='red')
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_roc(good_errors, bad_errors):
    y_true = [0] * len(good_errors) + [1] * len(bad_errors)
    y_scores = good_errors + bad_errors

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Autoencoder Anomaly Detection')
    plt.legend()
    plt.grid(True)
    plt.show()



def compute_per_image_error(inputs, outputs, alpha=0.84):
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    
    mse = torch.mean((inputs - outputs) ** 2, dim=(1, 2, 3))  # shape: (N,)
    ssim_val = ssim(inputs, outputs, data_range=1.0, size_average=False)  # shape: (N,)
    ssim_error = 1 - ssim_val

    error = (1 - alpha) * mse + alpha * ssim_error
    return error.cpu().numpy()


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1024, 1024)).astype(np.float32) / 255.0
    img = img[np.newaxis, np.newaxis, :, :]  # shape: (1,1,1024,1024)
    return img


def hundretmodeltest():
    model_name = "pytorchmodel_v2.onnx"
    dir_good = os.path.abspath("../data/screw/test/good")
    dir_bad = os.path.abspath("../data/screw/test/manipulated_front")

    good_paths = [os.path.join(dir_good, f) for f in os.listdir(dir_good) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    bad_paths = [os.path.join(dir_bad, f) for f in os.listdir(dir_bad) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    session = ort.InferenceSession(model_name)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    good_errors = []
    bad_errors = []

    for path in good_paths:
        img = preprocess_image(path)
        rec = session.run([output_name], {input_name: img})[0]
        error = compute_per_image_error(img, rec)
        good_errors.append(error[0])

    for path in bad_paths:
        img = preprocess_image(path)
        rec = session.run([output_name], {input_name: img})[0]
        error = compute_per_image_error(img, rec)
        bad_errors.append(error[0])


    plot_histograms(good_errors,bad_errors)
    # Labels: 0 = good, 1 = bad
    all_errors = np.concatenate([good_errors, bad_errors])
    all_labels = np.concatenate([np.zeros(len(good_errors)), np.ones(len(bad_errors))])

    fpr, tpr, thresholds = roc_curve(all_labels, all_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
hundretmodeltest()

#convert()

#session = ort.InferenceSession("model.onnx")

# Create dummy input
#dummy_input = np.random.rand(1, 1024, 1024, 1).astype(np.float32)

# Run inference
#input_name = session.get_inputs()[0].name
#outputs = session.run(None, {input_name: dummy_input})

#print("Inference output shape:", outputs[0].shape)

#test_inference()