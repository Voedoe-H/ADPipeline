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