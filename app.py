import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model_path = 'DenseNet201_1920_NL_Best.h5'
model = tf.keras.models.load_model(model_path)
# Set the image size
SIZE = 224

# Define the function to preprocess the image
def preprocess_image(img):
    img = img.resize((SIZE, SIZE))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Define the function to make predictions
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    return prediction

# Main application code
def main():
    st.title("Fundus Image Classifier")
    st.write("Upload an image and the app will predict if it's healthy or not.")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('Classifying...'):
            prediction = predict_image(image)

        if prediction[0][0] > 0.5:
            st.write("Prediction: Not Healthy")
        else:
            st.write("Prediction: Healthy")

if __name__ == "__main__":
    main()
