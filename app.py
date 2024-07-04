import streamlit as st
from PIL import Image
import sys

# Add the path to sys.path temporarily to import predict_step
sys.path.append("C:/Users/Prakriti Aayansh/OneDrive/Desktop/ImageTalk/Model")

# Import predict_step from imageCap.py
from imageCap import predict_step

# Center align the content
st.title("Hi, I am PixleBot 🤖")
st.subheader("I can make your images talk!")
st.markdown("---")  # Horizontal line for separation

# Center align the robot image
col1, col2, col3 = st.columns([1, 1, 1])  # Create three columns of equal width
with col2:  # Use the middle column for center alignment
    st.image("https://static.vecteezy.com/system/resources/previews/010/265/390/original/cute-3d-robot-say-hello-png.png", width=300)


st.title("To check my powers , upload the image below")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=200)
    st.write("")
    st.text("Generating caption...")

    # Save uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict caption using the image captioning model
    image_path = "temp_image.png"  # Path to the temporary image
    predictions = predict_step([image_path])

    # Display the predicted caption
    for idx, pred in enumerate(predictions):
        st.write(f"Image {idx+1} Caption: {pred}")
