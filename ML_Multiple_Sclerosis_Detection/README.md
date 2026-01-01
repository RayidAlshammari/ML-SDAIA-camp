# Multiple Sclerosis Detection Using MRI and Machine Learning 🧠

An AI-powered system for automated detection of Multiple Sclerosis (MS) from MRI brain scans using advanced machine learning techniques.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

---

##  Problem Statement

Multiple Sclerosis (MS) is a chronic neurological disease affecting the central nervous system. Detecting MS lesions from MRI scans is:
- Time-consuming and requires expert radiologists
- Subject to human error and variability
- Limited by availability of specialists

This project aims to **automate MS detection** using machine learning to assist healthcare professionals with faster and more accurate diagnostics.

---

##  Solution

An end-to-end machine learning system that:
-  Classifies MRI brain images as **MS (disease)** or **Healthy**
-  Achieves high accuracy using ensemble learning methods
-  Provides comprehensive analysis and model comparison
-  Displays detailed evaluation metrics and visualizations

---

##  Dataset

- **Source:** [Multiple Sclerosis MRI Dataset - Kaggle](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)
- **Total Images:** 3,427 labeled MRI brain scans
- **Classes:**
  - Healthy (Control): Axial & Sagittal views
  - MS (Disease): Axial & Sagittal views
- **Format:** Grayscale MRI images
- **Resolution:** Resized to 224×224 pixels

---

##  Technical Approach

### Feature Extraction

Two complementary feature extraction techniques:

1. **LBP (Local Binary Pattern)**
   - Captures texture patterns in MRI scans
   - Creates histogram of local binary patterns
   - Effective for identifying tissue abnormalities

2. **HOG (Histogram of Oriented Gradients)**
   - Captures shape and edge information
   - Detects structural changes in brain tissue
   - Robust to variations in lighting/contrast

### Data Processing Pipeline

```
Raw MRI Images (3,427 images)
    ↓
Preprocessing (Resize to 224×224, Grayscale)
    ↓
Feature Extraction (LBP + HOG → 6,094 features)
    ↓
Feature Scaling (StandardScaler)
    ↓
Dimensionality Reduction (PCA → 970 features, 95% variance)
    ↓
Data Split: Train (70%) / Validation (15%) / Test (15%)
    ↓
Model Training & Hyperparameter Tuning
    ↓
Validation & Test Evaluation
```

### Machine Learning Models

Three models were trained and compared:

1. **K-Nearest Neighbors (KNN)**
   - Hyperparameter tuning with GridSearchCV
   - Optimized: k, weights, distance metric

2. **Support Vector Machine (SVM)**
   - RBF kernel with hyperparameter tuning
   - Optimized: C, gamma, kernel type
   - **Selected as best individual model**

3. **Voting Classifier (Ensemble)**
   - Soft voting ensemble of KNN + SVM
   - Combines strengths of both models
   - Provides robust predictions

### Evaluation Metrics

**Performance Metrics:**
- Accuracy Score (Train / Validation / Test)
- Confusion Matrix with detailed breakdown (TP, TN, FP, FN)
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve & AUC Score
- False Positive Rate (FPR) & False Negative Rate (FNR)

**Validation Methods:**
- 10-Fold Cross-Validation
- Learning Curves (Overfitting Detection)
- Train/Validation/Test Split Evaluation

**Error Analysis:**
- Visual inspection of misclassified images
- Error pattern analysis across models
- Hard cases identification (images all models fail on)

---

## 📂 Project Structure

```
ML_Multiple_Sclerosis_Detection/
│
├── data/                          # MRI image dataset
│   ├── Control Axial_crop/       # Healthy - Axial view
│   ├── Control Saggital_crop/    # Healthy - Sagittal view
│   ├── MS Axial_crop/            # MS - Axial view
│   └── MS Saggital_crop/         # MS - Sagittal view
│
├── models/                        # Trained models (generated after running notebook)
│   ├── knn_model.pkl             # Trained KNN model
│   ├── svm_model.pkl             # Trained SVM model
│   ├── voting_model.pkl          # Voting Classifier
│   ├── scaler.pkl                # Feature scaler
│   └── pca.pkl                   # PCA transformer
│
├── app.ipynb                      # Complete ML pipeline & analysis
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore file
└── README.md                      # Project documentation (this file)
```

---

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Installation

1. **Clone the repository**
   ```bash
   https://github.com/RayidAlshammari/ML_Multiple_Sclerosis_Detection-
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis)
   - Extract to `data/` directory

---

##  Usage

### Run the Jupyter Notebook

Launch the notebook to train models and view analysis:

```bash
jupyter notebook app.ipynb
```

The notebook contains **11 main sections** with detailed subsections:

### 1. Import Libraries
   - All ML and visualization dependencies

### 2. Load Dataset
   - Load 3,427 MRI images from 4 folders
   - Resize to 224×224 and convert to grayscale

### 3. Exploratory Data Analysis
   - Class distribution visualization
   - Sample MRI images display

### 4. Feature Extraction
   - LBP: 10 texture features
   - HOG: 6,084 shape/edge features
   - Combine: 6,094 total features
   - Visualize extraction process

### 5. Feature Scaling
   - StandardScaler normalization
   - Before/After visualization

### 6. Dimensionality Reduction (PCA)
   - Scree plot analysis
   - Apply PCA (6,094 → 970 features, 95% variance)

### 7. Data Splitting
   - **Train: 70%** (2,399 images)
   - **Validation: 15%** (514 images)
   - **Test: 15%** (514 images)

### 8. Model Training & Evaluation
   - **KNN:** GridSearchCV → Validation → Test → Confusion Matrix
   - **SVM:** GridSearchCV → Validation → Test → Confusion Matrix
   - **Voting:** Ensemble → Validation → Test → Confusion Matrix
   - **Comparison:** Train/Valid/Test performance graphs

### 9. Advanced Analysis
   - **10-Fold Cross-Validation** on entire dataset
   - **Learning Curves** for overfitting detection
   - **ROC Curves** with AUC scores

### 10. Detailed Error Analysis
   - **Confusion Matrix Breakdown:** TP, TN, FP, FN for each model
   - **Misclassified Images:** Visual inspection of errors
   - **Hard Cases:** Images all models fail on

### 11. Save Models
   - Export trained models (KNN, SVM, Voting)
   - Save preprocessors (Scaler, PCA)

---

##  Results

### Data Split
- **Training Set:** 70% (2,399 images)
- **Validation Set:** 15% (514 images)
- **Test Set:** 15% (514 images)

### Model Performance

| Model | Train | Validation | Test | 10-Fold CV |
|-------|-------|------------|------|------------|
| **KNN** | ~94-96% | ~93-95% | ~93-95% | ~91% ± 2% |
| **SVM** | ~96-98% | ~94-96% | ~94-96% | ~92% ± 2% |
| **Voting Classifier** | ~95-97% | ~94-96% | ~95-96% | ~92% ± 2% |

*Note: Exact metrics vary based on random seed and hyperparameters*

### Error Analysis Summary

**False Positive (FP):** Healthy patients wrongly classified as MS
**False Negative (FN):** MS patients wrongly classified as Healthy (CRITICAL!)

All models maintain:
- **Low False Negative Rate** (critical for medical diagnosis)
- **Good balance** between FPR and FNR
- **Minimal overfitting** (validated via learning curves)

### Key Findings

- **Feature Extraction:** LBP + HOG combination captures both texture and structure
- **Dimensionality Reduction:** PCA reduces 84% of dimensions (6,094 → 970) while keeping 95% variance
- **Best Individual Model:** SVM with RBF kernel
- **Best Overall:** Voting Classifier for most robust and consistent predictions
- **Generalization:** Learning curves show good generalization (minimal overfitting)
- **Hard Cases:** ~1-2% of images are challenging for all models (require expert review)

---

##  Technologies Used

- **Python 3.12:** Core programming language
- **scikit-learn:** Machine learning models and evaluation
- **OpenCV:** Image processing
- **scikit-image:** Feature extraction (LBP, HOG)
- **NumPy & Pandas:** Data manipulation
- **Matplotlib & Seaborn:** Statistical visualization
- **Plotly:** Interactive charts
- **Jupyter Notebook:** Interactive development environment

---


##  Notebook Structure

The `app.ipynb` notebook is organized as follows:

### 1. Import Libraries
   - All required dependencies (scikit-learn, OpenCV, scikit-image, etc.)

### 2. Load Dataset
   - Load 3,427 MRI images from 4 folders
   - Preprocessing and resizing to 224×224

### 3. Exploratory Data Analysis
   - **3.1 Class Distribution:** Visualize balance between Healthy and MS classes
   - **3.2 Sample Images:** Display representative MRI samples

### 4. Feature Extraction
   - **4.1 LBP Features:** Extract 10 texture-based features
   - **4.2 HOG Features:** Extract 6,084 edge/shape features
   - **4.3 Combine Features:** Merge LBP + HOG (6,094 total features)
   - **4.4 Visualize Features:** Show LBP and HOG extraction process

### 5. Feature Scaling
   - StandardScaler normalization
   - Before/After visualization

### 6. Dimensionality Reduction (PCA)
   - **6.1 Variance Analysis:** Scree plot to determine optimal components
   - **6.2 Apply PCA (95% Variance):** Reduce to 970 features

### 7. Data Splitting (Train / Validation / Test)
   - Train: 70% (2,399 images)
   - Validation: 15% (514 images)
   - Test: 15% (514 images)

### 8. Model Training & Evaluation
   - **8.1 KNN Model:**
     - Hyperparameter tuning (GridSearchCV)
     - Validation set evaluation
     - Test set evaluation
     - Confusion Matrix

   - **8.2 SVM Model:**
     - Hyperparameter tuning (GridSearchCV)
     - Validation set evaluation
     - Test set evaluation
     - Confusion Matrix

   - **8.3 Voting Classifier:**
     - Ensemble training
     - Validation set evaluation
     - Test set evaluation
     - Confusion Matrix

   - **8.4 Model Comparison (Train / Valid / Test):**
     - Performance comparison across all splits

### 9. Advanced Analysis
   - **9.1 Cross-Validation (10-Fold):** Robust performance estimation
   - **9.2 Learning Curves (Overfitting Detection):** Analyze generalization
   - **9.3 ROC Curves:** AUC comparison across models

### 10. Detailed Error Analysis
   - **10.1 Confusion Matrix Breakdown (FP, FN, TP, TN):**
     - Detailed error analysis for each model
     - FPR vs FNR comparison

   - **10.2 Visualize Misclassified Images:**
     - Display False Positives (Healthy → MS)
     - Display False Negatives (MS → Healthy)

   - **10.4 Common Hard Cases:**
     - Identify images all models fail on
     - Error pattern analysis

### 11. Save Models
   - Export all trained models (KNN, SVM, Voting)
   - Save preprocessors (Scaler, PCA)

---

##  Features

### Comprehensive Analysis
- **Data Visualization:** Sample images, class distribution, feature distributions
- **Train/Valid/Test Split:** Proper model evaluation with 3-way split
- **Model Comparison:** Performance across Train, Validation, and Test sets
- **Confusion Matrices:** Detailed breakdown with TP, TN, FP, FN
- **ROC Curves:** AUC scores for all models
- **Classification Reports:** Precision, recall, F1-scores per class

### Advanced Validation
- **10-Fold Cross-Validation:** Robust performance estimation
- **Learning Curves:** Overfitting detection and model generalization analysis
- **Error Analysis:** Visual inspection of misclassified images
- **Hard Cases Identification:** Images that challenge all models

### Medical Diagnosis Focus
- **FPR vs FNR Analysis:** Critical for medical applications
- **False Negatives Tracking:** Identifying missed MS cases (most critical)
- **Error Pattern Analysis:** Understanding where and why models fail
- **Visual Error Inspection:** See actual misclassified MRI images

### Reproducibility
- **Fixed Random Seeds:** Consistent results across runs
- **Saved Models:** Reusable trained models
- **Complete Pipeline:** End-to-end workflow in one organized notebook
- **No Code Duplication:** Clean, structured implementation

--- 
## 👨‍💻 Author

**Rayid Alshammari**
Data Scientist & AI Engineer

---