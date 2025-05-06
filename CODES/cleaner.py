import os
import json
from bs4 import BeautifulSoup

# Directories
DATA_DIR = "./owncloud_docs"  # Directory containing HTML files
OUTPUT_DIR = "./processed_data"  # Directory for parsed data
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_html_text(html_path):
    """Extract clean text from HTML files."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove JavaScript, CSS, and unwanted tags
    for script in soup(["script", "style", "meta", "noscript"]):
        script.extract()

    # Extract and clean text
    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text


def save_to_json(data, output_path):
    """Save extracted data to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def process_html_files(data_dir, output_dir):
    """Process all HTML files and extract clean text."""
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                content = extract_html_text(file_path)

                # Prepare structured data
                data = {
                    "file_name": file,
                    "file_path": file_path,
                    "content": content,
                }

                # Save extracted content as JSON
                output_path = os.path.join(output_dir, f"{file}.json")
                save_to_json(data, output_path)


# Run the extraction
process_html_files(DATA_DIR, OUTPUT_DIR)
print("✅ HTML data extraction complete! Check the 'processed_data' folder.")
