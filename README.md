# Machine Learning Projects - Week 3
## SDAIA Bootcamp | Machine Learning Week

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)

**Week 3 Projects - Machine Learning Week**
**SDAIA AI Bootcamp**

</div>

---

##  Overview

This repository contains **three comprehensive projects** developed during **Week 3** (Machine Learning Week) of the SDAIA AI Bootcamp. The projects cover diverse ML domains including medical image classification, multi-label classification, and noise removal using various machine learning techniques.

---

## 📂 Project Structure

```
ML-SDAIA-camp/
│
├── 📁 ML_Multiple_Sclerosis_Detection/    # Project 1: MS Detection
│   ├── data/                               # MRI image dataset
│   │   ├── Control Axial_crop/            # Healthy - Axial view
│   │   ├── Control Saggital_crop/         # Healthy - Sagittal view
│   │   ├── MS Axial_crop/                 # MS - Axial view
│   │   └── MS Saggital_crop/              # MS - Sagittal view
│   ├── models/                             # Trained models
│   │   ├── knn_model.pkl                  # KNN model
│   │   ├── svm_model.pkl                  # SVM model
│   │   ├── voting_model.pkl               # Voting Classifier
│   │   ├── scaler.pkl                     # StandardScaler
│   │   └── pca.pkl                        # PCA Transformer
│   ├── app.ipynb                          # Main notebook
│   ├── requirements.txt                   # Dependencies
│   └── README.md                          # Project documentation
│
├── 📁 ml-challenge-Rayid-alshammari/      # Project 2: ML Challenge
│   ├── Task_1.ipynb                       # Task 1: Multi-Label Classification
│   ├── Task_2.ipynb                       # Task 2: Noise Removal
│   └── README.md                          # Project documentation
│
├── 📁 ml-classification-Rayid-alshammari/ # Project 3: MNIST Classification
│   ├── ML_Task_1_finale (1).ipynb         # Main classification notebook
│   └── README.md                          # Project documentation
│
├── .venv/                                  # Virtual environment (excluded from Git)
├── .gitignore                             # Git ignore file (all projects)
├── requirements.txt                       # Python dependencies (all projects)
└── README.md                              # This file - Comprehensive documentation
```

---

## Projects Overview

### 1️⃣ **Multiple Sclerosis Detection Using MRI**
**Automated MS detection from brain MRI scans**

<table>
<tr>
<td width="50%">

#### Description
An AI-powered system for automated detection of Multiple Sclerosis (MS) from MRI brain scans using advanced machine learning techniques.

#### Technologies Used
- **Feature Extraction**: LBP + HOG
- **Models**: KNN, SVM, Voting Classifier
- **Preprocessing**: StandardScaler, PCA
- **Dataset**: 3,427 MRI images

#### Performance
- **KNN**: ~93-94% accuracy
- **SVM**: ~94-96% accuracy
- **Voting Classifier**: ~94-96% accuracy ⭐

</td>
<td width="50%">

#### 📁 Main Files

**app.ipynb** (Main Notebook)
- Load dataset (3,427 images)
- Feature extraction (LBP + HOG → 6,094 features)
- Train 3 different models
- Comprehensive evaluation with Confusion Matrix
- Cross Validation & Learning Curves
- ROC Curves & Error Analysis
- Save trained models

**models/** (Saved Models)
- All trained models ready for deployment
- Preprocessors (Scaler, PCA) saved

**requirements.txt** (in project root)
- All dependencies listed in main `requirements.txt`
- Includes scikit-learn, opencv-python, scikit-image
- Plus visualization libraries (matplotlib, seaborn, plotly)

</td>
</tr>
</table>

**Key Features:**
-  Complete pipeline from raw images to predictions
-  3-way split: Train (70%) / Validation (15%) / Test (15%)
-  Advanced error analysis with FP/FN breakdown
-  Learning curves for overfitting detection
-  Visual inspection of misclassified images

---

### **Machine Learning Challenge**
**Two-task ML challenge: Multi-label classification & Noise removal**

<table>
<tr>
<td width="50%">

#### Task 1: Multi-Label Classification
**Simultaneous prediction of multiple labels**

**Objective**: Classify MNIST digits into two categories simultaneously:
- ✓ **Odd or Not** - Is the digit odd?
- ✓ **Greater Than 5** - Is the digit > 5?

**Models Used**:
- Logistic Regression (~90% accuracy)
- Random Forest (~97-98% accuracy) ⭐

**File**: `Task_1.ipynb`

**Pipeline**:
1. Load MNIST dataset (70,000 images)
2. Convert labels to multi-label format
3. Train MultiOutputClassifier models
4. Evaluate performance for each output separately

**Results**:
- Both labels predicted with high accuracy
- Random Forest significantly outperforms Logistic Regression
- Demonstrates effective multi-label classification

</td>
<td width="50%">

#### Task 2: Noise Removal
**ML-based image denoising**

**Objective**: Remove noise from MNIST images and reconstruct clean versions

**Technique Used**:
- KNN Regressor for reconstruction
- Gaussian noise (factor: 0.5)
- Distance-weighted predictions

**File**: `Task_2.ipynb`

**Pipeline**:
1. Add random Gaussian noise to images
2. Train KNN Regressor on noisy-clean pairs
3. Reconstruct clean images from noisy inputs
4. Evaluate using MSE, MAE, R² metrics

**Performance**:
- **R² Score**: 0.7751
- **RMSE**: 37.2
- **MAE**: 13.37
- **Classification Accuracy**: 74.89%

**Includes**:
- Visual comparison: Noisy → Denoised → Original
- Comprehensive regression metrics

</td>
</tr>
</table>

---

### 3️⃣ **MNIST Digit Classification**
**Classic handwritten digit classification with model comparison**

<table>
<tr>
<td width="50%">

####  Description
A comprehensive classification project on the MNIST dataset (0-9 handwritten digits) featuring three different models with detailed performance comparison.

#### Models Implemented

**1. K-Nearest Neighbors (KNN)**
- n_neighbors: 3
- algorithm: auto
- **Accuracy**: 97%
- **Prediction Time**: 30.73 seconds

**2. Random Forest Classifier** ⭐ **Best Overall**
- n_estimators: 100
- **Accuracy**: 97%
- **Prediction Time**: 0.40 seconds
- **Best balance** of speed and accuracy

**3. Logistic Regression**
- max_iter: 1000
- **Accuracy**: 92%
- **Prediction Time**: 0.04 seconds
- **Fastest** but lower accuracy

</td>
<td width="50%">

#### 📁 Main File

**ML_Task_1_finale (1).ipynb**

**Contents**:
- Load MNIST dataset (70,000 images, 784 features)
- Data split (80% train, 20% test)
- Train 3 different classifiers
- Comprehensive performance comparison:
  - Accuracy scores
  - Classification reports
  - Prediction time analysis
  - Precision, Recall, F1-Score per class

**Detailed Metrics**:
- Per-class performance (digits 0-9)
- Confusion matrix analysis
- Speed vs Accuracy trade-offs

**Key Findings**:
- **Random Forest**: Best choice for production (fast + accurate)
- **Logistic Regression**: Best for real-time applications (fastest)
- **KNN**: Accurate but too slow for large-scale deployment

</td>
</tr>
</table>

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- Virtual environment (recommended)

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd ML-SDAIA-camp

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# For Mac/Linux:
source .venv/bin/activate
# For Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running the Projects

#### **Project 1: Multiple Sclerosis Detection**

```bash
cd ML_Multiple_Sclerosis_Detection

# Download dataset first from Kaggle
# https://www.kaggle.com/datasets/buraktaci/multiple-sclerosis
# Extract to data/ directory

# Launch notebook
jupyter notebook app.ipynb

# Run all cells sequentially
# Models will be saved to models/ directory
```

#### **Project 2: ML Challenge**

```bash
cd ml-challenge-Rayid-alshammari

# Task 1: Multi-Label Classification
jupyter notebook Task_1.ipynb

# Task 2: Noise Removal
jupyter notebook Task_2.ipynb

# Dataset (MNIST) downloads automatically via scikit-learn
```

#### **Project 3: MNIST Classification**

```bash
cd ml-classification-Rayid-alshammari

# Launch notebook
jupyter notebook "ML_Task_1_finale (1).ipynb"

# Dataset downloads automatically via scikit-learn
```

---

## Required Libraries

All dependencies are consolidated in the main `requirements.txt` file in the project root.

### Installation

```bash
# Navigate to project root
cd ML-SDAIA-camp

# Install all dependencies at once
pip install -r requirements.txt
```

### Libraries Included

**Core Data Science:**
- numpy==1.26.4
- pandas==2.2.1
- matplotlib==3.8.3
- seaborn==0.13.2

**Machine Learning:**
- scikit-learn==1.4.1.post1

**Deep Learning:**
- tensorflow>=2.13.0 (for Project 2 - Task 2)
- keras>=2.13.0

**Image Processing:**
- opencv-python==4.9.0.80 (for Project 1)
- scikit-image==0.22.0 (for Project 1)
- pillow==10.2.0

**Visualization:**
- plotly==5.19.0

**Jupyter:**
- jupyter==1.0.0
- notebook==7.1.0
- ipykernel>=6.20.0

---

## Learning Outcomes

### Machine Learning Concepts

- ✅ **Feature Engineering**: LBP (texture), HOG (edges)
- ✅ **Dimensionality Reduction**: PCA (6,094 → 970 features, 95% variance)
- ✅ **Classification Algorithms**: KNN, SVM, Random Forest, Logistic Regression
- ✅ **Ensemble Methods**: Voting Classifier (soft voting)
- ✅ **Multi-Label Learning**: MultiOutputClassifier
- ✅ **Regression for Denoising**: KNN Regressor
- ✅ **Hyperparameter Tuning**: GridSearchCV
- ✅ **Model Evaluation**: Accuracy, Precision, Recall, F1-Score

### Advanced Techniques

-  **Cross-Validation**: 10-fold CV for robust evaluation
-  **Learning Curves**: Detecting overfitting/underfitting
-  **ROC Analysis**: AUC scores for model comparison
-  **Error Analysis**: Confusion matrix, FP/FN rates
-  **Model Comparison**: Speed vs Accuracy trade-offs
-  **Model Persistence**: Saving/loading trained models with pickle

### Practical Skills

-  **Data Preprocessing**: Scaling, normalization, noise injection
-  **Data Visualization**: Professional plots with matplotlib/seaborn
-  **Image Processing**: OpenCV, scikit-image
-  **Experimental Design**: Train/Valid/Test splits
-  **Documentation**: Writing clear, comprehensive READMEs

---

## Results Summary

### Performance Overview

| Project | Best Model | Accuracy | Special Notes |
|---------|-----------|----------|---------------|
| **MS Detection** | Voting Classifier | 94-96% | ✅ Low False Negatives (critical for medical) |
| **Multi-Label (Task 1)** | Random Forest | 97-98% | ✅ Both labels predicted accurately |
| **Noise Removal (Task 2)** | KNN Regressor | R²=0.78 | ✅ Good reconstruction quality |
| **MNIST Classification** | Random Forest | 97% | ✅ Fast inference (0.40s) |

### Detailed Metrics

#### Project 1: MS Detection
- **Training Set**: 2,398 images (70%)
- **Validation Set**: 514 images (15%)
- **Test Set**: 515 images (15%)
- **10-Fold CV**: 92% ± 2%
- **ROC AUC**: 0.97-0.98
- **False Negative Rate**: ~11% (minimized for medical safety)

#### Project 2: Multi-Label Classification
- **Label 1 (Odd/Not)**: 98% precision, 98% recall
- **Label 2 (Greater Than 5)**: 97% precision, 97% recall
- **Overall Performance**: Excellent for both tasks

#### Project 3: MNIST Comparison
- **KNN**: High accuracy but slow (30s prediction time)
- **Random Forest**: Best balance (97% accuracy, 0.4s)
- **Logistic Regression**: Fastest (0.04s) but 92% accuracy

---

## Key Insights

### Medical Image Classification (Project 1)
- **Feature Engineering is Critical**: LBP + HOG combination captures both texture and structure
- **PCA Reduces Complexity**: 84% dimension reduction while keeping 95% variance
- **Ensemble Methods Excel**: Voting classifier provides most robust predictions
- **Error Analysis Matters**: Visual inspection revealed challenging edge cases
- **Medical Context**: Low FNR is more important than low FPR (don't miss sick patients)

### Multi-Label Learning (Project 2, Task 1)
- **Random Forest Dominates**: Significantly outperforms Logistic Regression
- **Multi-Label is Effective**: Can predict multiple targets simultaneously
- **Task Correlation**: Both tasks achieve similar high performance

### Noise Removal (Project 2, Task 2)
- **KNN Works for Regression**: Distance-weighted neighbors reconstruct clean images
- **Trade-off Exists**: High R² but moderate pixel-level accuracy
- **Visual Quality**: Denoised images are visually recognizable despite imperfect metrics

### Model Selection (Project 3)
- **Speed vs Accuracy**: Random Forest offers best balance
- **Application Matters**: Choose model based on deployment constraints
- **MNIST is Well-Solved**: All models achieve >90% accuracy

---

##  Technical Challenges & Solutions

### Challenge 1: Large Feature Dimensionality
**Problem**: 6,094 features from LBP + HOG extraction
**Solution**: PCA dimensionality reduction (95% variance retained)
**Impact**: Faster training, reduced overfitting, maintained accuracy

### Challenge 2: Class Imbalance Awareness
**Problem**: Medical diagnosis requires minimizing false negatives
**Solution**: Detailed FP/FN analysis, focus on FNR metric
**Impact**: More appropriate model evaluation for medical context

### Challenge 3: Convergence Warnings
**Problem**: Logistic Regression convergence warnings
**Solution**: Documented but acceptable (doesn't affect final performance)
**Alternative**: Could increase max_iter or add feature scaling

### Challenge 4: Slow KNN Predictions
**Problem**: KNN takes 30s for predictions (too slow)
**Solution**: Random Forest as production alternative
**Impact**: 75x speedup with same accuracy

---

## 👨‍💻 Author

**Rayid Alshammari**
Data Scientist & AI Engineer

---

<div align="center">

**🎯 SDAIA AI Bootcamp - Week 3**
**Machine Learning Week**
**Three comprehensive projects demonstrating ML fundamentals**

⭐ **Star this repository if you found it helpful!** ⭐

</div>
