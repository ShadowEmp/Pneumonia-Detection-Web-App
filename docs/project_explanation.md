# Project Explanation: Pneumonia Detection System

This document provides a comprehensive technical deep-dive into the Pneumonia Detection System. It covers the architecture, machine learning pipeline, deployment strategy, and configuration details.

## 1. High-Level Overview

The system is a full-stack AI application designed to assist medical professionals in screening for pneumonia using chest X-rays. It leverages a split-stack architecture:

### A. The Backend (`app.py`, `model.py`)
Built with **Flask**, serving as the inference engine.
*   **API Endpoints:**
    *   `/api/predict`: Fast inference, returns JSON prediction.
    *   `/api/predict-with-gradcam`: Detailed inference, returns prediction + heatmap images.
    *   `/api/health`: Health check for monitoring uptime.
*   **Libraries:** `tensorflow` (AI), `opencv-python` (Image Processing), `flask-cors` (Cross-Origin Resource Sharing).

### B. The Frontend (`UploadPage.jsx`)
Built with **React** and **Tailwind CSS**.
*   **Interactive UI:** Uses `framer-motion` for smooth state transitions (Idle -> Scanning -> Result).
*   **PDF Generation:** `jspdf` creates professional medical reports client-side, including the AI's findings and the heatmap image.
*   **Proxying:** Vite is configured to proxy API requests to the backend, avoiding CORS issues during development.

### C. Configuration (`config.py`)
The system behavior is governed by central configuration:
*   **Image Dimensions:** `224x224x3` (Standard ResNet input).
*   **Prediction Threshold:** `0.3` (Sensitivity-biased). This means any probability > 30% is flagged as Pneumonia. This is intentional for a screening tool to minimize false negatives (missing a sick patient).
*   **Data Augmentation:** During training, images are rotated (20°), zoomed (20%), and shifted to make the model robust against different X-ray alignments.

---

## 3. The Machine Learning Model (The "Brain")

### Architecture: Transfer Learning with ResNet50
We utilize **Transfer Learning**, leveraging a pre-trained **ResNet50** model.
*   **Base Model:** ResNet50 (trained on ImageNet). We use it as a "feature extractor" to understand low-level visual patterns (edges, textures).
*   **Custom Head:** We attach a custom classifier to the base:
    1.  **GlobalAveragePooling2D:** Flattens the 3D feature map.
    2.  **Dense (512, ReLU):** Learns high-level combinations of features.
    3.  **Dropout (0.5):** Randomly drops 50% of connections during training to prevent overfitting.
    4.  **Dense (1, Sigmoid):** Outputs a single probability score.

### Training Strategy
*   **Optimizer:** Adam (`learning_rate=0.0001`).
*   **Loss Function:** Binary Crossentropy (standard for Yes/No classification).
*   **Metrics:** Accuracy, Precision, Recall, and AUC (Area Under Curve).
*   **Callbacks:**
The backend is packaged using **Docker** to ensure consistency across environments.
*   **Base Image:** `python:3.10`
*   **System Deps:** Installs `libgl1` (required for OpenCV).
*   **Setup:** Installs Python dependencies from `requirements.txt`.
*   **Startup:** Runs `download_model.py` to fetch the large model file from Google Drive, then starts the Gunicorn server.

### Hosting Strategy
*   **Backend:** Hosted on **Hugging Face Spaces**.
    *   Uses the Dockerfile to build the environment.
    *   Exposes port `7860`.
*   **Frontend:** Hosted on **Vercel**.
    *   Connects to the backend via the configured API URL.
    *   Serves the static React assets globally via CDN.

### Model Management
*   The model file (`best_pneumonia_model.h5`) is too large for Git.
*   **Solution:** It is hosted externally (Google Drive). The `download_model.py` script automatically fetches it when the Docker container starts, ensuring the app always has the latest model without bloating the repository.
