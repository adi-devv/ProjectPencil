import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import os

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Create output folder
os.makedirs("words", exist_ok=True)


def ocr_on_image(pil_img):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


# Load image in grayscale
img_path = "img.png"  # your image file
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Threshold to binary inverted image
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Dilate to merge letters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 4))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1], b[0]))

# OCR and save each word image
full_text = ""
for idx, (x, y, w, h) in enumerate(bounding_boxes):
    word_img = img[y:y + h, x:x + w]
    word_img_inv = 255 - word_img
    pil_img = Image.fromarray(word_img_inv)

    # Save image to words folder
    save_path = os.path.join("words", f"word_{idx+1}.png")
    pil_img.save(save_path)

    # Run OCR
    text = ocr_on_image(pil_img)
    full_text += text + " "

print("Extracted Text:")
print(full_text.strip())
