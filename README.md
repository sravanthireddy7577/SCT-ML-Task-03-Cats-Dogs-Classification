# 🐱🐶 Cats vs Dogs Classification using SVM

## About

This project is part of my **Machine Learning Internship at SkillCraft Technology**.

The objective is to classify images into two categories: **Cat** and **Dog** using a **Support Vector Machine (SVM)**.

The project includes image preprocessing, HOG feature extraction, PCA dimensionality reduction, SVM classification, and a Streamlit web application.

## 🚀 Features

- Cat and Dog image classification
- Image resizing and grayscale conversion
- HOG feature extraction
- Feature scaling using StandardScaler
- PCA dimensionality reduction
- SVM classification with RBF kernel
- Streamlit web application
- Image upload and prediction

## 📊 Dataset

The dataset contains:

- **8,000 training images**
- **2,000 testing images**
- **2 classes:** Cats and Dogs

Images are resized to **64 × 64 pixels** before feature extraction.

The dataset is not included in the GitHub repository because of its large size.

## 🤖 Machine Learning Model

The project uses:

- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** RBF
- **C:** 10
- **Feature Extraction:** HOG
- **Scaling:** StandardScaler
- **Dimensionality Reduction:** PCA

### Workflow

```text
Input Image
     ↓
Resize to 64 × 64
     ↓
Grayscale Conversion
     ↓
HOG Feature Extraction
     ↓
Feature Scaling
     ↓
PCA
     ↓
SVM Classifier
     ↓
Cat / Dog Prediction
```

## 📁 Project Structure

```text
├── app.py
├── train_model.py
├── svm_model.pkl
├── scaler.pkl
├── pca.pkl
├── requirements.txt
├── .gitignore
├── .gitattributes
└── README.md
```

The `dataset/` folder is stored locally and is not uploaded to GitHub.

## 🛠️ Installation and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Make sure the dataset is available locally in:

```text
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
    ├── cats/
    └── dogs/
```

Run:

```bash
python train_model.py
```

This creates:

```text
svm_model.pkl
scaler.pkl
pca.pkl
```

### 3. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### 4. Make a Prediction

- Upload a JPG, JPEG, or PNG image.
- The image is preprocessed.
- HOG features are extracted.
- The trained model predicts **Cat** or **Dog**.

## ⚠️ Limitations

- The model is designed for Cat and Dog classification.
- Images containing both a cat and a dog may produce a single prediction.
- Images from other categories may still be classified as either Cat or Dog.
- The model uses grayscale images.

## 💼 Internship Details

**Organization:** SkillCraft Technology  
**Domain:** Machine Learning  
**Task:** Task 03 – Cats vs Dogs Classification using SVM

## 👩‍💻 Author

**Sravanthi**

B.Tech – Computer Science and Engineering
