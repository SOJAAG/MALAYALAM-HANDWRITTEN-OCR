import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

CATEGORIES = ["മ", "ഭ", "ന", "ക", "വ", "റ", "ഒ"]


def preprocessing(input_image, edge=False, inv_thresh=False):
    gray_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    if inv_thresh:
        ret, im_th = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        im_th = cv2.adaptiveThreshold(
            im_th, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 10
        )
        im_th = cv2.bitwise_not(im_th)
    else:
        ret, im_th = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        im_th = cv2.adaptiveThreshold(
            im_th, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10
        )
    if edge:
        edge_image = cv2.Canny(im_th, 0, 255)
        return edge_image
    return im_th


@st.cache_resource
def load_my_model():
    model = load_model("weights 96.h5")
    return model


model = load_my_model()


st.title("Image Classification App")
st.write("Upload an image and the model will predict the category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Convert PIL Image to OpenCV format
    img_array = np.array(image.convert("RGB"))
    img_array = img_array[:, :, ::-1].copy()  # Convert RGB to BGR

    # Preprocess the image
    processed_image = preprocessing(img_array)  # Use your preprocessing function
    resized_image = cv2.resize(processed_image, (28, 28))
    input_image = resized_image.reshape(-1, 28, 28, 1).astype(float) / 255.0

    # Make a prediction
    prediction = model.predict(input_image)
    predicted_category_index = np.argmax(prediction, axis=1)[0]
    predicted_category = CATEGORIES[predicted_category_index]

    # st.write(f"Prediction: {predicted_category}")
    st.title(f"Prediction: {predicted_category}")
    # st.markdown(
    # f"<h2 style='text-align: center; color: #4CAF50;'>Prediction: {predicted_category}</h2>",
    # unsafe_allow_html=True,
    # )
