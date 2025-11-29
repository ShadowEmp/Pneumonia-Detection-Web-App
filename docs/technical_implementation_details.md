# Technical Implementation Details

## 1. Implementation Summary
*   **Tech Stack:** React (Frontend), Flask (Backend), TensorFlow/Keras (AI), Docker (Containerization).
*   **Model:** ResNet50 (Transfer Learning) customized for binary classification (Normal vs. Pneumonia).
*   **Key Features:** Drag-and-drop upload, real-time AI analysis, Grad-CAM visual explanation, PDF report generation.
*   **Deployment:** Split-stack architecture with Vercel (Frontend) and Hugging Face Spaces (Backend).

## 2. Deep Dive: Matrices & CNNs

### A. Images as Matrices
To a computer, an image is just a giant grid of numbers (a **Matrix**).
*   **Grayscale Image:** A 2D matrix (Height x Width). Each cell (pixel) is a number from 0 (Black) to 255 (White).
*   **Color Image (RGB):** A 3D matrix (Height x Width x 3). It has three layers (channels): Red, Green, and Blue.

**Example:** A 224x224 X-ray image is actually a matrix of size `224 x 224 x 3`. That's `150,528` individual numbers that the model must process!

### B. How CNNs Work (Convolutional Neural Networks)
A CNN is a special type of AI designed to process these image matrices.

#### 1. The Convolution Operation (The Filter)
Imagine a small flashlight (a **Filter** or **Kernel**) shining over the top-left corner of the image.
*   The filter is a small matrix (e.g., 3x3) of numbers.
*   It multiplies its numbers with the image's pixel numbers underneath it.
*   It sums up the result to get a single number.
*   It then slides (convolves) one step to the right and repeats.

**Why?** This process detects features. One filter might be designed to find vertical edges. Another might find curves. Another might find textures.

#### 2. Pooling (Downsampling)
After convolution, the image is still large. **Pooling** shrinks it to make processing faster and to focus on the most important features.
*   **Max Pooling:** Takes a 2x2 block of pixels and keeps only the highest number. It discards the rest.
*   **Result:** The image gets smaller, but the important features (like a bright spot indicating pneumonia) are preserved.

#### 3. Flattening & Fully Connected Layers
*   **Flattening:** After many layers of Convolution and Pooling, we have a small, deep 3D matrix of high-level features (e.g., "cloudy lung texture"). We flatten this into a single long list (vector) of numbers.
*   **Fully Connected (Dense) Layer:** This is a traditional neural network. It looks at this long list and decides: *'Given these features (cloudiness, edges), what is the probability of Pneumonia?'*

### Summary of the Flow
`Image Matrix (Pixels)` -> `Convolution (Find Edges)` -> `Pooling (Shrink)` -> `Convolution (Find Shapes)` -> `Pooling` -> `Flatten` -> `Dense Layer (Decision)` -> `Prediction (Pneumonia/Normal)`
