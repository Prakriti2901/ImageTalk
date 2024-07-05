import streamlit as st
from PIL import Image
import sys

# Adding the path
sys.path.append("C:/Users/Prakriti Aayansh/OneDrive/Desktop/ImageTalk/Model")

# Importing function from imageCap file
from imageCap import predict_step

# Page config
st.set_page_config(
    page_title="PixleBot",
    page_icon="🤖",
    layout="centered",
)

# Hiding the main menu and footer
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-15vlnr2.e8zbici0 {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hi, I am PixleBot 🤖")
st.subheader("I can make your images talk!")
st.markdown("---")  # Horizontal line for separation

# Align the robot image
col1, col2, col3 = st.columns([1, 1, 1])  # Creating three columns of equal width
with col2:  # Using middle column for alignment
    st.image("https://static.vecteezy.com/system/resources/previews/010/265/390/original/cute-3d-robot-say-hello-png.png", width=300)

st.title("To check my powers, upload the image below")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=200)
    st.write("")
    st.text("Generating caption...")

    # Saving uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predicting caption using the model
    image_path = "temp_image.png"  # Path to the temporary image
    predictions = predict_step([image_path])

    # Displaying caption
    for idx, pred in enumerate(predictions):
        st.write(f"Image {idx+1} Caption: {pred}")
