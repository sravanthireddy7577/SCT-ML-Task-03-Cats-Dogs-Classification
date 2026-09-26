import cv2
import os
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------
# SETTINGS
# -----------------------------------

IMG_SIZE = 64

TRAIN_PATH = "dataset/training_set"
TEST_PATH = "dataset/test_set"


# -----------------------------------
# LOAD DATA + HOG FEATURES
# -----------------------------------

def load_data(folder):

    features = []
    labels = []

    for label, category in enumerate(["cats", "dogs"]):

        path = os.path.join(folder, category)

        print("Reading:", path)

        for file in os.listdir(path):

            img_path = os.path.join(path, file)

            img = cv2.imread(img_path)

            if img is not None:

                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Convert to grayscale
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # HOG feature extraction
                feature = hog(
                    img,
                    orientations=9,
                    pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2),
                    block_norm="L2-Hys"
                )

                features.append(feature)
                labels.append(label)

    return np.array(features), np.array(labels)


# -----------------------------------
# LOAD TRAINING DATA
# -----------------------------------

print("\nLoading training data...")

X_train, y_train = load_data(TRAIN_PATH)

print("Training data shape:", X_train.shape)


# -----------------------------------
# LOAD TEST DATA
# -----------------------------------

print("\nLoading test data...")

X_test, y_test = load_data(TEST_PATH)

print("Testing data shape:", X_test.shape)


# -----------------------------------
# SCALE FEATURES
# -----------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------------
# PCA
# -----------------------------------

print("\nApplying PCA...")

pca = PCA(
    n_components=300,
    svd_solver="randomized",
    random_state=42
)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print("Features after PCA:", X_train.shape)


# -----------------------------------
# TRAIN SVM
# -----------------------------------

print("\nTraining SVM model...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

print("SVM training completed!")


# -----------------------------------
# PREDICTION
# -----------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# -----------------------------------
# ACCURACY
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

print("\nConfusion Matrix:")

print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# SAVE FILES
# -----------------------------------

print("\nSaving model files...")

joblib.dump(model, "svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model saved successfully!")

print("\nTask 03 completed successfully!")