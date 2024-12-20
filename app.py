import streamlit as st
import cv2
import numpy as np
from modelscope.pipelines import pipeline
from PIL import Image
import tempfile

# Load the pipeline for portrait matting
portrait_matting = pipeline('portrait-matting')

def process_image(input_image_path):
    """
    Process the image using the portrait matting pipeline and return the result image path.
    """
    result = portrait_matting(input_image_path)
    output_img = result['output_img']
    return output_img

def main():
    # Streamlit App Title
    st.title("UI ONE")
    st.write("Upload an image and get the portrait matting result.")

    # Upload image widget
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Convert uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_input:
            temp_input.write(uploaded_file.read())
            input_image_path = temp_input.name
        
        # Display the uploaded image
        st.subheader("Uploaded Image")
        uploaded_image = Image.open(input_image_path)
        st.image(uploaded_image, caption="Input Image", use_column_width=True)
        
        # Process the image
        st.subheader("Processing...")
        output_img = process_image(input_image_path)
        
        # Save the result to a temporary file for display
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_output:
            cv2.imwrite(temp_output.name, output_img)
            result_image_path = temp_output.name

        # Display the result image
        st.subheader("Result Image")
        result_image = Image.open(result_image_path)
        st.image(result_image, caption="Portrait Matting Result", use_column_width=True)

if __name__ == "__main__":
    main()
