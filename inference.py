# Imports
import os
from PIL import Image
import torch
from torchvision import transforms
from models.birefnet import BiRefNet
from utils import check_state_dict

# Load BiRefNet Model and Weights
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load('BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.pth', map_location=device)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)
birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

# Transformation for Input Image
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Process an Image from a Local Path
def process_image(image_path, output_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure it's in RGB mode
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

    # Save the output image
    image_masked.save(output_path, format="PNG")
    print(f"Processed image saved at: {output_path}")

# Paths
input_image_path = "Screenshot 2025-01-11 162833.jpg"  # Replace with your input image path
output_image_path = "processed.jpg"  # Replace with your output path

# Process and Save Image
process_image(input_image_path, output_image_path)
