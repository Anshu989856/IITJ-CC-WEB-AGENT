import os
import json
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

# Download NLTK data if not already downloaded
nltk.download("punkt")

# Directories
input_dir = "./pdfsprocessed"     # Directory containing the processed JSON files
output_dir = "./semantic_chunks"   # Directory where chunked files will be stored
os.makedirs(output_dir, exist_ok=True)

# Load the Sentence Transformer model (all-MiniLM-L6-v2 is fast and accurate)
model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_text(text):
    """
    Split the text into sentences and filter out very short sentences.
    """
    sentences = sent_tokenize(text)
    # Filter out sentences with less than 4 words to avoid meaningless chunks
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    return sentences

def cluster_sentences(sentences, num_clusters=5):
    """
    Encode sentences into embeddings and cluster them with KMeans.
    Returns cluster labels and embeddings.
    """
    embeddings = model.encode(sentences)
    # Adjust the number of clusters if there are fewer sentences than desired clusters
    n_clusters = min(num_clusters, len(sentences))
    if n_clusters < 1:
        return [], embeddings
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, embeddings

def merge_cluster_sentences(sentences, labels):
    """
    Merge sentences belonging to the same cluster in the original order.
    Returns a dictionary mapping clusters to merged content.
    """
    cluster_map = {}
    for idx, label in enumerate(labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append((idx, sentences[idx]))

    merged = {}
    for label, items in cluster_map.items():
        # Sort sentences by original order to maintain context
        sorted_items = [s for idx, s in sorted(items, key=lambda x: x[0])]
        merged[label] = " ".join(sorted_items)
    return merged

def process_file(input_path, output_path, num_clusters=5):
    """
    Process a single JSON file to create semantic chunks.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    file_name = data.get("file_name", "unknown")
    content = data.get("content", "")

    # Preprocess: split content into sentences
    sentences = preprocess_text(content)
    if not sentences:
        print(f"⚠️ No meaningful content found in {file_name}.")
        return
    
    # Cluster sentences semantically
    labels, embeddings = cluster_sentences(sentences, num_clusters=num_clusters)
    merged_chunks = merge_cluster_sentences(sentences, labels)
    
    # Prepare output chunks
    chunks = []
    for label, chunk_text in merged_chunks.items():
        chunks.append({
            "cluster": int(label),
            "chunk_content": chunk_text
        })

    output_data = {
        "file_name": file_name,
        "file_path": data.get("file_path", ""),
        "chunks": chunks
    }

    # Save the semantic chunks to a new JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Processed {file_name}: Created {len(chunks)} semantic chunks.")

# Process all JSON files in the input directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(".json"):
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace(".json", "_semantic_chunked.json"))
        # Adjust num_clusters dynamically based on content length
        process_file(input_path, output_path, num_clusters=5)
