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

## 📈 Model Performance

The trained HOG + PCA + SVM model achieved an accuracy of **77.0%** on the test dataset.

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| Cat | 0.77 | 0.78 | 0.77 |
| Dog | 0.77 | 0.76 | 0.77 |

**Overall Accuracy: 77.0%**

### Confusion Matrix

```text
[[777 223]
 [237 763]]
```

The model correctly classified:

- **777 Cat images**
- **763 Dog images**

## 🔄 Workflow

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

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Scikit-image
- Joblib
- Pillow
- Streamlit
- Git
- GitHub

## ⚙️ Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/sravanthireddy7577/SCT-ML-Task-03-Cats-Dogs-Classification.git
```

### 2. Navigate to the Project Folder

```bash
cd SCT-ML-Task-03-Cats-Dogs-Classification
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

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

### 5. Run the Streamlit Application

```bash
python -m streamlit run app.py
```

### 6. Make a Prediction

- Upload a JPG, JPEG, or PNG image.
- The image is preprocessed.
- HOG features are extracted.
- The features are scaled.
- PCA is applied.
- The trained SVM model predicts **Cat** or **Dog**.

## 🌐 Live Application

The project is deployed using **Streamlit Community Cloud**.

The application allows users to upload an image and receive a Cat or Dog prediction.

## ⚠️ Limitations

- The model is designed for Cat and Dog classification.
- Images containing both a cat and a dog may produce a single prediction.
- Images from other categories may still be classified as either Cat or Dog.
- The model uses grayscale images.
- Prediction performance may vary depending on image quality, lighting, background, and similarity to the training dataset.

## 💼 Internship Details

**Organization:** SkillCraft Technology

**Domain:** Machine Learning

**Task:** Task 03 – Cats vs Dogs Classification using SVM

## 👩‍💻 Author

**Sravanthi**

B.Tech – Computer Science and Engineering
