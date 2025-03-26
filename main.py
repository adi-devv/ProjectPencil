import torch
from torchvision import transforms
from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np

# Load your handwritten page
img = Image.open("my_handwriting.jpg").convert("L")  # Grayscale
# Preprocess (example: crop to a word)
word_img = img.crop((50, 50, 150, 100))  # Adjust coordinates
word_img.save("target.png")

# Create source image (typed version)
text = "hello"  # Transcribe what’s in word_img
source_img = Image.new("L", (100, 50), 255)  # White background
draw = ImageDraw.Draw(source_img)
font = ImageFont.truetype("arial.ttf", 20)
draw.text((10, 10), text, font=font, fill=0)  # Black text
source_img.save("source.png")

# Pix2Pix setup (simplified)
# Use pretrained Pix2Pix from a repo, e.g., https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# Fine-tune with your pairs: source.png → target.png

# Inference (after training)
def generate_handwriting(text, generator):
    source = Image.new("L", (len(text)*20, 50), 255)
    draw = ImageDraw.Draw(source)
    draw.text((10, 10), text, font=font, fill=0)
    source_tensor = transforms.ToTensor()(source).unsqueeze(0)
    with torch.no_grad():
        output = generator(source_tensor)
    return transforms.ToPILImage()(output.squeeze())

# Example usage (after training)
new_text = "Hi there"
output_img = generate_handwriting(new_text, generator)  # Generator from trained model
output_img.save("output.png")