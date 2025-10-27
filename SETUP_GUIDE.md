# Complete Setup Guide - Pneumonia Detection System

This guide will walk you through setting up the entire pneumonia detection system from scratch.

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8 or higher installed
- [ ] Node.js 16 or higher installed
- [ ] Git installed
- [ ] At least 10GB free disk space
- [ ] Internet connection for downloading dependencies
- [ ] (Optional) GPU with CUDA support for faster training

## 🔧 Step-by-Step Setup

### Step 1: Environment Setup

#### 1.1 Create Project Directory
```bash
mkdir pneumonia-detection
cd pneumonia-detection
```

#### 1.2 Set Up Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Install Backend Dependencies

#### 2.1 Install Python Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.15
- Flask and Flask-CORS
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- And other dependencies

#### 2.2 Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

### Step 3: Download Dataset

#### Option A: Kaggle Dataset (Recommended)

1. **Install Kaggle CLI:**
```bash
pip install kaggle
```

2. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Download `kaggle.json`
   - Place it in:
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - Linux/Mac: `~/.kaggle/kaggle.json`

3. **Download dataset:**
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

4. **Extract dataset:**
```bash
# Windows (using PowerShell)
Expand-Archive chest-xray-pneumonia.zip -DestinationPath data

# Linux/Mac
unzip chest-xray-pneumonia.zip -d data
```

#### Option B: Manual Download

1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download"
3. Extract to `data/` folder

### Step 4: Organize Dataset Structure

Ensure your data folder looks like this:
```
data/
├── train/
│   ├── NORMAL/
│   │   ├── image1.jpeg
│   │   ├── image2.jpeg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── image1.jpeg
│       ├── image2.jpeg
│       └── ...
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Note:** The dataset folder names might be uppercase. Rename them to match:
```bash
# Windows
ren data\train\NORMAL data\train\Normal
ren data\train\PNEUMONIA data\train\Pneumonia

# Linux/Mac
mv data/train/NORMAL data/train/Normal
mv data/train/PNEUMONIA data/train/Pneumonia
```

### Step 5: Train the Model

#### 5.1 Quick Training (Default Settings)
```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Build ResNet50 model
- Train for 30 epochs
- Save model to `models/pneumonia_model.h5`
- Generate visualizations in `results/`

#### 5.2 Custom Training

Edit `train.py` to customize:
```python
# Line ~150
MODEL_TYPE = 'resnet50'  # or 'vgg16', 'custom_cnn'
EPOCHS = 30
BATCH_SIZE = 32
FINE_TUNE = True
```

#### 5.3 Monitor Training

Training will display:
```
Epoch 1/30
157/157 [==============================] - 45s 287ms/step - loss: 0.3245 - accuracy: 0.8567
```

Expected training time:
- **With GPU**: 30-45 minutes
- **CPU only**: 2-4 hours

### Step 6: Verify Model Training

After training completes, check:
```bash
# Windows
dir models
dir results

# Linux/Mac
ls models/
ls results/
```

You should see:
- `models/pneumonia_model.h5`
- `models/best_pneumonia_model.h5`
- `results/training_history.png`
- `results/confusion_matrix.png`
- `results/roc_curve.png`

### Step 7: Set Up Frontend

#### 7.1 Navigate to Frontend Directory
```bash
cd frontend
```

#### 7.2 Install Node Dependencies
```bash
npm install
```

This will install:
- React and React Router
- Vite build tool
- Tailwind CSS
- Axios, Recharts, Lucide icons
- And other dependencies

#### 7.3 Verify Installation
```bash
npm list react
```

### Step 8: Test the Backend API

#### 8.1 Start Flask Server

In the project root (not frontend):
```bash
# Make sure you're in the main directory
cd ..  # if you're still in frontend

# Start the server
python app.py
```

You should see:
```
Starting Pneumonia Detection API...
Model loaded from models/best_pneumonia_model.h5
 * Running on http://0.0.0.0:5000
```

#### 8.2 Test API Endpoints

Open a new terminal and test:
```bash
# Health check
curl http://localhost:5000/api/health

# Model info
curl http://localhost:5000/api/model-info
```

### Step 9: Start Frontend Development Server

#### 9.1 Open New Terminal

Keep the Flask server running, open a new terminal:
```bash
cd frontend
npm run dev
```

You should see:
```
VITE v5.0.8  ready in 523 ms

➜  Local:   http://localhost:3000/
➜  Network: use --host to expose
```

#### 9.2 Access the Application

Open your browser and navigate to:
```
http://localhost:3000
```

### Step 10: Test the Complete System

#### 10.1 Upload Test Image

1. Go to the Upload page
2. Drag and drop a chest X-ray image
3. Click "Analyze Image"
4. View prediction results and Grad-CAM visualization

#### 10.2 Test Sample Images

Use images from `data/test/` folder for testing.

## 🎯 Quick Start Commands (Summary)

After initial setup, use these commands to run the system:

**Terminal 1 - Backend:**
```bash
cd miniproj5
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd miniproj5/frontend
npm run dev
```

**Access:** http://localhost:3000

## 🐛 Troubleshooting

### Issue: TensorFlow Installation Failed

**Solution:**
```bash
# Try installing specific version
pip install tensorflow==2.15.0

# Or use CPU-only version
pip install tensorflow-cpu==2.15.0
```

### Issue: Port 5000 Already in Use

**Solution:**
Edit `config.py`:
```python
API_PORT = 5001  # Change to different port
```

### Issue: Model Not Found Error

**Solution:**
```bash
# Ensure model exists
ls models/

# If missing, train the model
python train.py
```

### Issue: Frontend Build Errors

**Solution:**
```bash
# Clear node_modules and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Issue: CORS Errors

**Solution:**
Ensure Flask-CORS is installed:
```bash
pip install flask-cors
```

### Issue: Out of Memory During Training

**Solution:**
Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Issue: Slow Training on CPU

**Solution:**
- Reduce number of epochs
- Use smaller model (custom_cnn instead of resnet50)
- Consider using Google Colab with free GPU

## 📊 Expected Results

After successful setup:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: ~96%
- **Prediction Time**: < 2 seconds per image
- **Grad-CAM Generation**: < 1 second

## 🚀 Production Deployment

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t pneumonia-detection .
docker run -p 5000:5000 pneumonia-detection
```

### Deploy to Cloud

**Heroku:**
```bash
heroku create pneumonia-detection-app
git push heroku main
```

**AWS/Azure/GCP:**
Follow respective cloud provider documentation for Flask app deployment.

## 📝 Next Steps

1. ✅ Complete setup and training
2. ✅ Test with sample images
3. ✅ Review model performance
4. 📊 Analyze results in Analysis page
5. 🎨 Customize UI if needed
6. 🚀 Deploy to production (optional)

## 💡 Tips for Best Results

1. **Data Quality**: Use high-quality, properly labeled X-ray images
2. **Training Time**: Let the model train completely (don't interrupt)
3. **GPU Usage**: Use GPU for 10x faster training
4. **Fine-tuning**: Enable fine-tuning for better accuracy
5. **Validation**: Always validate predictions with medical professionals

## 📞 Need Help?

If you encounter issues:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Check Python and Node.js versions
5. Refer to README.md for additional information

---

**Setup Complete! 🎉**

You now have a fully functional AI-powered pneumonia detection system!
