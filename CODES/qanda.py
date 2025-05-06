import json
import os
import tiktoken
from llama_cpp import Llama
import chromadb

# Paths
JSON_DIR = "./processed_data"  # Directory where JSON files are stored
MODEL_PATH = "./models/llama-2-7b.Q4_0.gguf"  # Path to LLaMA 2 model
OUTPUT_DIR = "./qna_results"  # Output directory for generated Q&A
CHUNK_SIZE = 512  # Optimal chunk size (tokens)
OVERLAP = 50  # Token overlap between chunks

# Initialize LLaMA 2 Model
llama = Llama(model_path=MODEL_PATH, n_ctx=2048)

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chromadb_storage")
collection = chroma_client.get_or_create_collection(name="qna_owncloud")

# Create Output Directory if not Exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tokenizer for Chunking (LLaMA 2 uses 'cl100k_base' tokenizer)
enc = tiktoken.get_encoding("cl100k_base")


# Function to split content into overlapping chunks
def chunk_content(content, chunk_size, overlap):
    tokens = enc.encode(content)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks


# Function to generate Q&A pairs for a chunk
def generate_qa(chunk):
    prompt = f"Generate relevant questions and answers based on the following text:\n\n{chunk}\n\nQuestions and Answers:"
    output = llama(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["\n\n"],
    )
    qa_text = output["choices"][0]["text"].strip()
    return qa_text


# Process all JSON files in the directory
for filename in os.listdir(JSON_DIR):
    if filename.endswith(".json"):
        file_path = os.path.join(JSON_DIR, filename)

        # Read the JSON content
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract content directly
        if "content" in data:
            all_content = data["content"]
        else:
            print(f"⚠️ Skipping {filename} - No content found.")
            continue

        # Chunk the content before passing to LLaMA
        chunks = chunk_content(all_content, CHUNK_SIZE, OVERLAP)

        # Store all generated Q&A
        all_qa_pairs = []
        for i, chunk in enumerate(chunks):
            qa_output = generate_qa(chunk)
            all_qa_pairs.append({"chunk_id": i + 1, "qa_pairs": qa_output})

            # Add to ChromaDB for retrieval
            collection.add(
                documents=[qa_output],
                metadatas=[{"file_name": filename, "chunk_id": i + 1}],
                ids=[f"{filename.replace('.json', '')}_chunk_{i+1}"],
            )

        # Save Q&A pairs to JSON
        output_file_path = os.path.join(OUTPUT_DIR, f"{filename.replace('.json', '_qna.json')}")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump({"file_name": filename, "qa_pairs": all_qa_pairs}, f, indent=4)

        print(f"✅ Q&A generated and stored for: {filename}")

print("🎉 Q&A generation with chunking completed for all files!")
