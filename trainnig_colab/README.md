# PneumoVision Training Scripts

This directory contains scripts to fine-tune the Pneumonia Detection model using Google Colab.

## 📂 Files
*   `extract.py`: Unzips and organizes the dataset (removes Covid class, keeps Normal/Pneumonia).
*   `dataset.py`: Handles data loading and augmentation.
*   `train.py`: Fine-tunes the ResNet50 model.
*   `run_colab.sh`: Helper script to run the pipeline.

## 🚀 Usage in Colab

1.  Upload this folder to your Google Drive.
2.  Upload the `best_pneumonia_model.h5` (base model) and your dataset zip file.
3.  Run the scripts to generate a new, fine-tuned `finetuned_pneumonia_model.h5`.
