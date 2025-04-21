# Anomaly Detection with MVTec AD Dataset

This repository provides a full pipeline for anomaly detection using the [MVTec Anomaly Detection (AD)](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) dataset. The project covers everything from data preprocessing, model training/testing with PyTorch, to plans for deployment via a REST API in a Docker container.

## Dataset

The MVTec AD dataset is a high-quality dataset for benchmarking industrial anomaly detection methods. It contains a variety of object and texture categories, with both normal and anomalous images.

Download it from: [Kaggle - MVTec AD](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)

Place the dataset in a suitable directory (e.g., `./data/raw/`) before running preprocessing.

## Preprocessing

The preprocessing step prepares the MVTec AD dataset for model training. To run the preprocessing script:

```bash
python data/preprocess.py
```

## Model

The model can be found in model/pytorch_model.py. Im using a simple CNN Autoencoder approach. For a loss i utilized a combination of classic Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM) loss.

### Model Performance

The performances here are with a model that was trained for only 100 epochs of the training loop defined in the model script.

#### Reconstruction Error - Scratch Head
![Reconstruction Error - Scratch Head](docs/scratch_head_errors.png)

#### Reconstruction Error - Scratch Neck
![Reconstruction Error - Scratch Neck](docs/scratch_neck_errors.png)

#### Reconstruction Error - Thread Side
![Reconstruction Error - Thread Side](docs/thread_side_errors.png)

#### Reconstruction Error - Thread Top
![Reconstruction Error - Thread Top](docs/thread_top_errors.png)

| Defect Type          | Mean Error | Std Dev | Min Error | Max Error |
|----------------------|------------|---------|-----------|-----------|
| Scratch Head         | 0.0620     | 0.0015  | 0.0580    | 0.0652    |
| Scratch Neck         | 0.0606     | 0.0013  | 0.0574    | 0.0630    |
| Thread Side          | 0.0611     | 0.0018  | 0.0590    | 0.0663    |
| Thread Top           | 0.0613     | 0.0015  | 0.0591    | 0.0646    |