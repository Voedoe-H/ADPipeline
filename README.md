# InsightAD: Anomaly Detection with MVTec AD Dataset

This repository provides a full pipeline for anomaly detection using the [MVTec Anomaly Detection (AD)](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) dataset. The project covers everything from data preprocessing, model training/testing with PyTorch, to plans for deployment via a REST API in a Docker container.

## Dataset

The MVTec AD dataset is a high-quality dataset for benchmarking industrial anomaly detection methods. It contains a variety of object and texture categories, with both normal and anomalous images.

Download it from: [Kaggle - MVTec AD](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)

Place the dataset in a suitable directory (e.g., `./data/raw/`) before running preprocessing.

## Preprocessing

The preprocessing script augments the "good" screw images from the MVTec AD dataset to increase the dataset size for training. It applies the following transformations:

* Random rotations (±40°)
* Shifting (±20%)
* Shearing (±20%)
* Zooming (±20%)
* Horizontal flipping

Usage:

```bash
python data/preprocess.py
```

## Model

### Architecture Overview

The core model is based on a Convolutional Autoencoder architecture. It consists of an encoder-decoder structure aimed at reconstructing the input image while learning efficient feature representations of normal (non-anomalous) images.

* **Encoder:** 
    The encoder is composed of four convolutional layers. Each layer reduces the spatial dimensions of the input image while increasing the depth (number of feature maps). The encoder extracts high-level features from the image, which will later be used for reconstruction.
    * Conv1: Convolutional layer with 32 output channels.
    * Conv2: Convolutional layer with 64 output channels.
    * Conv3: Convolutional layer with 128 output channels.
    * Conv4: Convolutional layer with 256 output channels.
    
    After each convolutional layer, a Batch Normalization layer is applied to stabilize and speed up training, followed by a ReLU activation function to introduce non-linearity.

* **Decoder:**
    The decoder consists of four transpose convolutional layers that gradually increase the spatial dimensions to match the original input size. It is designed to reconstruct the input image from the feature representations produced by the encoder.
    * DecoderConv2: Transpose convolution with 128 output channels.
    * DecoderConv3: Transpose convolution with 64 output channels.
    * DecoderConv4: Transpose convolution with 32 output channels.
    * DecoderConv5: Final layer producing the output with 1 channel (grayscale image) using a Sigmoid activation function.

    The decoder uses Batch Normalization and ReLU activation to maintain stability and enhance training performance.

### Loss Function
The model is trained using a hybrid loss function combining two critical components:
1. **Mean Squared Error(MSE):**
    Measures pixel-wise errors between the reconstructed and original images.

2. **Structural Similarity Index Measure (SSIM):**
    A perceptual loss that considers the structural similarity between the input and the reconstructed images. This helps the model focus not only on pixel-level accuracy but also on maintaining high-level structural features.

The total loss is computed as a weighted sum of MSE and SSIM:
    
    Total Loss = (1-α) x MSE + α x (1-SSIM)

Where α is a hyperparameter controlling the relative weight of the two losses. By default, α = 0.84.
This hybrid loss function encourages the model to reconstruct both fine-grained details and the high-level structure of normal images, making it more sensitive to anomalous deviations.

### Model Performance

The trained model was evaluated using reconstruction error as the primary metric. An ROC analysis was used to select a threshold of 0.0595 for detecting anomalies, achieving a balance between true positive and false positive rates. Below are sample results from reconstruction error analysis for different defect types.

#### Reconstruction Error - Scratch Head
![Reconstruction Error - Scratch Head](docs/scratch_head_errors.png)

#### Reconstruction Error - Scratch Neck
![Reconstruction Error - Scratch Neck](docs/scratch_neck_errors.png)

#### Reconstruction Error - Thread Side
![Reconstruction Error - Thread Side](docs/thread_side_errors.png)

#### Reconstruction Error - Thread Top
![Reconstruction Error - Thread Top](docs/thread_top_errors.png)

#### ROC Analysis
![Reconstruction Error - Thread Top](docs/ROC.png)

| Defect Type          | Mean Error | Std Dev | Min Error | Max Error |
|----------------------|------------|---------|-----------|-----------|
| Scratch Head         | 0.0620     | 0.0015  | 0.0580    | 0.0652    |
| Scratch Neck         | 0.0606     | 0.0013  | 0.0574    | 0.0630    |
| Thread Side          | 0.0611     | 0.0018  | 0.0590    | 0.0663    |
| Thread Top           | 0.0613     | 0.0015  | 0.0591    | 0.0646    |

### Potential Improvements

Future improvements could include exploring more advanced architectures (e.g., UNet) or incorporating feature-level loss functions using pretrained networks (e.g., Perceptual Loss with VGG16) to improve the model's sensitivity to finer structural anomalies. Additionally, experimenting with data augmentation or transfer learning might enhance the model's ability to generalize to different types of defects.

## Deployment

The trained model can be deployed using a simple C++ HTTP server implementation. The server provides a single /infer POST endpoint for anomaly detection.

### Running the Inference Server

To build and run the server locally, follow these steps inside the deployment/ directory:

```bash
mkdir build
cd build
cmake ..
make
```

Once built, start the server:

```bash
./InferenceServer
```

By default, the server listens on 0.0.0.0:8080.
You can modify the listening address and port inside the server source code if needed.

The /infer POST endpoint expects a base64-encoded 1024x1024 image provided inside a JSON object under the key "image".
The server responds with a simple JSON containing a boolean value under the key "Anomaly", indicating whether an anomaly was detected.

### Running the Inference Server as a Docker Container

Alternatively, the inference server can be deployed as a Docker container.
To build the Docker image, execute the following inside the deployment/ directory:

```bash
docker build -t inference-server .
```

To run the image:

```bash
docker run -p 8080:8080 inference-server
```