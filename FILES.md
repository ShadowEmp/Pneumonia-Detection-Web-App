# Project Files

## Essential Files for Running the Application

### Core Application
- `app.py` - Flask backend API server
- `config.py` - Configuration settings
- `model.py` - Model loading and utilities
- `data_preprocessing.py` - Image preprocessing
- `gradcam_simple.py` - Grad-CAM visualization
- `requirements.txt` - Python dependencies

### Frontend
- `frontend/` - React web interface (12 files)

### Documentation
- `README.md` - Main project documentation
- `GETTING_STARTED.md` - Quick start guide
- `SETUP_GUIDE.md` - Detailed setup instructions
- `API_DOCUMENTATION.md` - API reference
- `PROJECT_SUMMARY.md` - Project overview
- `LICENSE` - MIT License

### Directories
- `data/` - Dataset directory (empty, add your data here)
- `models/` - Trained model files (add your .h5 model here)
- `uploads/` - Temporary upload directory
- `logs/` - Training logs
- `results/` - Results and visualizations

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Add your trained model to `models/` folder
3. Start backend: `python app.py`
4. Start frontend: `cd frontend && npm install && npm run dev`
5. Open browser: `http://localhost:5173`

## What Was Removed

- Training scripts (train.py, evaluation.py)
- Utility scripts (quick_start.py, predict_single.py)
- Old Grad-CAM implementation (gradcam.py)
- Colab training files (entire training_colab/ folder)
- Unnecessary documentation files
- Result directories from Colab
