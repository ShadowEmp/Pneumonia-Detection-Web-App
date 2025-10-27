# Dataset Information

## Chest X-Ray Images (Pneumonia) Dataset

### Overview
This project uses the Chest X-Ray Images (Pneumonia) dataset, which contains X-ray images organized into categories for training a pneumonia detection model.

### Dataset Source
- **Name**: Chest X-Ray Images (Pneumonia)
- **Source**: Kaggle
- **URL**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **License**: CC BY 4.0

### Dataset Description
The dataset contains 5,863 X-ray images (JPEG) organized into 2 categories:
- **Normal**: Healthy lungs
- **Pneumonia**: Lungs with pneumonia infection

### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

### Dataset Statistics

#### Training Set
- **Total Images**: 5,216
- **Normal**: 1,341 (25.7%)
- **Pneumonia**: 3,875 (74.3%)

#### Validation Set
- **Total Images**: 16
- **Normal**: 8 (50%)
- **Pneumonia**: 8 (50%)

#### Test Set
- **Total Images**: 624
- **Normal**: 234 (37.5%)
- **Pneumonia**: 390 (62.5%)

### Image Specifications
- **Format**: JPEG
- **Color**: Grayscale (converted to RGB for model input)
- **Original Size**: Variable (typically 1000-2000 pixels)
- **Model Input Size**: Resized to 224×224 pixels
- **Bit Depth**: 8-bit

### Data Source Details
The chest X-ray images were selected from retrospective cohorts of pediatric patients:
- Age: 1-5 years old
- Location: Guangzhou Women and Children's Medical Center, Guangzhou
- Time Period: Routine clinical care

### Image Quality Control
All chest X-ray imaging was performed as part of patients' routine clinical care. For the analysis:
- Low quality or unreadable scans were removed
- All images were reviewed by expert physicians
- Diagnoses were graded by two expert physicians
- A third expert checked the evaluation set to account for grading errors

### Class Imbalance
The dataset has class imbalance:
- Training set: ~74% Pneumonia, ~26% Normal
- Test set: ~62% Pneumonia, ~38% Normal

**Handling Strategy:**
- Data augmentation for minority class
- Class weights during training
- Stratified splitting
- Evaluation metrics: Precision, Recall, F1-Score (not just accuracy)

### Data Preprocessing

#### Applied Transformations
1. **Resizing**: All images resized to 224×224 pixels
2. **Normalization**: Pixel values scaled to [0, 1]
3. **Color Conversion**: Grayscale converted to RGB (3 channels)

#### Data Augmentation (Training Only)
- Rotation: ±20 degrees
- Width/Height Shift: ±20%
- Shear: ±20%
- Zoom: ±20%
- Horizontal Flip: Yes
- Fill Mode: Nearest

### Ethical Considerations

#### Privacy
- All images are de-identified
- No patient information included
- Compliant with HIPAA regulations

#### Bias Considerations
- Dataset from single institution (may have geographic bias)
- Pediatric patients only (1-5 years)
- May not generalize to adult populations
- Equipment and imaging protocols specific to source institution

### Citation
If you use this dataset, please cite:

```
Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), 
"Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification", 
Mendeley Data, V2, doi: 10.17632/rscbjbr9sj.2
```

### Related Publications
- Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172(5):1122-1131.e9.

### Download Instructions

#### Method 1: Kaggle CLI
```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

#### Method 2: Manual Download
1. Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (requires Kaggle account)
3. Extract to `data/` directory

### Data Organization for This Project
After downloading, organize as:
```
miniproj5/
└── data/
    ├── train/
    │   ├── Normal/
    │   └── Pneumonia/
    ├── val/
    │   ├── Normal/
    │   └── Pneumonia/
    └── test/
        ├── Normal/
        └── Pneumonia/
```

### Alternative Datasets
If you want to use different datasets:

1. **NIH Chest X-ray Dataset**
   - 112,120 X-ray images
   - 14 disease labels
   - https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

2. **CheXpert Dataset**
   - 224,316 chest radiographs
   - 14 observations
   - https://stanfordmlgroup.github.io/competitions/chexpert/

3. **MIMIC-CXR Database**
   - 377,110 images
   - 227,835 studies
   - https://physionet.org/content/mimic-cxr/2.0.0/

### Data Limitations

1. **Geographic Limitation**: Single institution in China
2. **Age Limitation**: Pediatric patients only (1-5 years)
3. **Temporal Limitation**: Specific time period
4. **Equipment Limitation**: Specific X-ray equipment
5. **Class Imbalance**: More pneumonia cases than normal
6. **Binary Classification**: Does not distinguish bacterial vs viral

### Recommendations for Use

✅ **Good For:**
- Research and educational purposes
- Proof-of-concept models
- Learning deep learning techniques
- Grad-CAM and explainability studies

⚠️ **Not Recommended For:**
- Direct clinical deployment without validation
- Adult patient diagnosis
- Real-time critical care decisions
- Replacing radiologist interpretation

### Data Quality Notes

**Strengths:**
- Expert-labeled by physicians
- Quality control performed
- Sufficient sample size for deep learning
- Publicly available

**Weaknesses:**
- Class imbalance
- Limited demographic diversity
- Single institution source
- Pediatric-only population

### Future Dataset Improvements

To improve model generalization:
1. Include multi-institutional data
2. Add adult patient X-rays
3. Balance class distribution
4. Include more demographic information
5. Add temporal validation data
6. Include different X-ray equipment types

---

**Last Updated**: 2024
**Dataset Version**: V2
