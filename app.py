import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from models.birefnet import BiRefNet
from utils import check_state_dict
import io

import warnings
warnings.filterwarnings("ignore")
# Load BiRefNet Model and Weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth', map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)
birefnet.to(device)
birefnet.eval()

# Transformation for Input Image
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Process Image Function
def process_image(image):
    # Preprocess the image
    input_image = transform_image(image).unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        preds = birefnet(input_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()

    # Convert prediction to PIL image
    pred_pil = transforms.ToPILImage()(pred)

    # Apply the mask to the original image
    image_masked = image.resize((1024, 1024))
    image_masked.putalpha(pred_pil)

    return image_masked

# Streamlit UI
st.title("BiRefNet Watermark Removal")

st.write("Upload an image, and the model will process it to remove the watermark.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Process the image
    processed_image = process_image(image)
    
    # Display processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    
    # Convert image to bytes and create a downloadable link
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    buffered.seek(0)
    
    st.download_button(
        label="Download Processed Image",
        data=buffered,
        file_name="processed_image.png",
        mime="image/png"
    )
