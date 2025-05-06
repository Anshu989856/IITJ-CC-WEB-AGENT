import os
import json
from PyPDF2 import PdfReader

# Folder containing PDFs
pdf_folder = './pdfs'
# Folder to save JSON files
json_folder = './pdfsprocessed'
os.makedirs(json_folder, exist_ok=True)

# Loop over all PDFs
for filename in os.listdir(pdf_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)

        # Extract all text
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

        # Create a dictionary
        data = {
            "filename": filename,
            "content": text.strip()
        }

        # Save as JSON
        json_filename = filename.replace('.pdf', '.json')
        json_path = os.path.join(json_folder, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

print("Conversion complete!")
