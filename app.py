import streamlit as st
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from PIL import Image


# -----------------------------------
# LOAD MODEL FILES
# -----------------------------------

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")


# -----------------------------------
# STREAMLIT PAGE
# -----------------------------------

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    page_icon="🐱",
    layout="centered"
)

st.title("🐱🐶 Cats vs Dogs Image Classifier")

st.write(
    "Upload an image and the trained SVM model "
    "will predict whether it is a Cat or a Dog."
)


# -----------------------------------
# IMAGE UPLOAD
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------------
# PREDICTION
# -----------------------------------

if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert PIL image to NumPy
    image = np.array(image)

    # Handle RGB / RGBA images
    if len(image.shape) == 3:

        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize
    image = cv2.resize(image, (64, 64))

    # -----------------------------------
    # HOG FEATURE EXTRACTION
    # -----------------------------------

    feature = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # Convert to 2D
    feature = feature.reshape(1, -1)

    # -----------------------------------
    # SCALE
    # -----------------------------------

    feature = scaler.transform(feature)

    # -----------------------------------
    # PCA
    # -----------------------------------

    feature = pca.transform(feature)

    # -----------------------------------
    # PREDICTION
    # -----------------------------------

    prediction = model.predict(feature)[0]

    # -----------------------------------
    # DISPLAY RESULT
    # -----------------------------------

    if prediction == 0:

        st.success("🐱 Prediction: CAT")

    else:

        st.success("🐶 Prediction: DOG")