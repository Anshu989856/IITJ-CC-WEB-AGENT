import os
import json
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# -------------------------------------------
# Configuration
# -------------------------------------------
CHUNK_DIR = "./semantic_chunks"        # Directory with chunked JSON files
CHROMA_DB_DIR = "./chroma_db"          # Directory to store ChromaDB

# -------------------------------------------
# Initialize Embedding Model & ChromaDB
# -------------------------------------------
# Use a local sentence transformer model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create ChromaDB instance
chroma_client = chromadb.PersistentClient(CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(name="owncloud_chunks")

# -------------------------------------------
# Function to Store Chunks in ChromaDB
# -------------------------------------------
def store_chunks_in_chromadb(file_path):
    """Store all chunks from a given JSON file into ChromaDB."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    file_name = data.get("file_name", "unknown")
    chunks = data.get("chunks", [])
    
    docs = []
    metadata_list = []
    
    for i, chunk in enumerate(chunks):
        chunk_content = chunk.get("chunk_content", "")
        if chunk_content.strip():
            # Create a document with chunk content
            doc = Document(
                page_content=chunk_content,
                metadata={"file_name": file_name, "chunk_id": i}
            )
            docs.append(doc)
            metadata_list.append({
                "file_name": file_name,
                "chunk_id": str(i)
            })
    
    if docs:
        # Embed and store documents in ChromaDB
        Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            collection_name="owncloud_chunks",
            persist_directory=CHROMA_DB_DIR
        )
        print(f"✅ Stored {len(docs)} chunks from '{file_name}' into ChromaDB.")

# -------------------------------------------
# Process All Chunked Files
# -------------------------------------------
def process_all_chunk_files(chunk_dir):
    """Process and store all chunked JSON files in the specified directory."""
    for file_name in os.listdir(chunk_dir):
        if file_name.endswith("_semantic_chunked.json"):
            file_path = os.path.join(chunk_dir, file_name)
            store_chunks_in_chromadb(file_path)

# -------------------------------------------
# Main Execution
# -------------------------------------------
if __name__ == "__main__":
    process_all_chunk_files(CHUNK_DIR)
    print("🎉 All chunks have been successfully embedded and stored in ChromaDB!")
