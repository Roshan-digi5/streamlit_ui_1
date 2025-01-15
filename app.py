import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import io

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoModelForImageSegmentation

# Enforce CPU-only execution
device = 'cpu'
torch.set_float32_matmul_precision('high')

# Load BiRefNet model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    'zhengpeng7/BiRefNet-DIS5K-TR_TEs',
    trust_remote_code=True
)
birefnet.to(device)
birefnet.eval()

# Transformation for input image
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to process the image
def process_image(image):
    input_image = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = birefnet(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    image_masked = image.resize((1024, 1024))
    image_masked.putalpha(pred_pil)
    return image_masked

# Streamlit app
st.title("BiRefNet Watermark Removal (CPU Only)")
st.write("Upload an image, and the model will process it to remove the watermark.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Validate file size
        if uploaded_file.size > 10 * 1024 * 1024:  # 10 MB limit
            st.error("Please upload an image smaller than 10 MB.")
            st.stop()

        # Load and display the original image
        image = Image.open(uploaded_file).convert("RGB")

        # Show progress spinner
        with st.spinner("Processing image..."):
            processed_image = process_image(image)

        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Convert processed image to bytes for download
        buffered = io.BytesIO()
        processed_image.save(buffered, format="PNG")
        buffered.seek(0)

        st.download_button(
            label="Download Processed Image",
            data=buffered,
            file_name="processed_image.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
