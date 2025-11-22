# PneumoVision: AI-Powered Pneumonia Detection

PneumoVision is a state-of-the-art web application designed to assist medical professionals and patients in the early detection of pneumonia from chest X-ray images. Powered by a fine-tuned ResNet50 deep learning model, it provides instant, accurate analysis with visual explainability using Grad-CAM technology.

![PneumoVision Hero](https://via.placeholder.com/800x400?text=PneumoVision+Interface)

## 🚀 Features

*   **AI Diagnosis:** High-accuracy detection of pneumonia using a ResNet50 convolutional neural network.
*   **Visual Explainability:** Grad-CAM heatmaps highlight the exact regions of the lung that influenced the AI's decision.
*   **Comprehensive Reports:** Generate detailed, medical-grade PDF reports with diagnosis, visual evidence, and actionable next steps.
*   **Modern UI/UX:** A futuristic, responsive interface built with React, Framer Motion, and Tailwind CSS.
*   **Secure & Private:** Client-side processing capabilities and secure backend handling.

## 🛠️ Tech Stack

*   **Frontend:** React, Vite, Tailwind CSS, Framer Motion, Lucide React, jsPDF
*   **Backend:** Python, Flask, TensorFlow/Keras, OpenCV, NumPy
*   **Model:** ResNet50 (Pre-trained & Fine-tuned)

## 📦 Installation

### Prerequisites
*   Node.js (v16+)
*   Python (v3.8+)

### Backend Setup
1.  Navigate to the root directory:
    ```bash
    cd pneumovision
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Model Setup:**
    *   Download the fine-tuned model `best_pneumonia_model.h5` (or `pneumonia_model.h5`) from your source.
    *   Place it in the `models/` directory.
    *   *Note: The model file is excluded from the repository due to size limits.*
5.  Start the Flask server:
    ```bash
    python app.py
    ```

### Frontend Setup
1.  Navigate to the frontend directory:
    ```bash
    cd frontend
    ```
2.  Install dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```

## 🏥 Usage

1.  Open the application in your browser (typically `http://localhost:5173`).
2.  Click "Initiate Scan" or drag and drop a chest X-ray image (JPEG/PNG).
3.  Wait for the Neural Scanner to analyze the image.
4.  View the diagnostic result, confidence score, and heatmap.
5.  Click "Download Full Report" to get a comprehensive PDF summary.

## ⚠️ Disclaimer

This tool is intended for **educational and screening assistance purposes only**. It is not a replacement for professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.
