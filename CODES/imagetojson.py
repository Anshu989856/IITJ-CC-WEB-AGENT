import os
import json
from PIL import Image
import pytesseract

# OPTIONAL: Tell where tesseract is installed (only if necessary)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Folders
image_folder = './images'  # <-- Change this
json_folder = './pdfsprocessed'           # <-- Change this
os.makedirs(json_folder, exist_ok=True)

# Loop through images
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(image_folder, filename)
        
        # Open the image
        img = Image.open(image_path)
        
        # Extract text using OCR
        text = pytesseract.image_to_string(img)
        
        # Create dictionary
        data = {
            "filename": filename,
            "content": text.strip()
        }
        
        # Save as JSON
        json_filename = filename.rsplit('.', 1)[0] + '.json'
        json_path = os.path.join(json_folder, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ All images converted to JSON!")
