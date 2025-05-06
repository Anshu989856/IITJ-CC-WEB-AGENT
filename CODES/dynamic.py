import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# --- Configurations ---
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "owncloud_chunks"

# --- Initialize components ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- Dynamic Helpers ---

def dynamic_top_k(query):
    """Adjust top_k based on query length."""
    length = len(query.split())
    if length <= 5:
        return 5
    elif length <= 15:
        return 50
    else:
        return 70

def dynamic_model_selection(query):
    """Select model dynamically based on query complexity."""
    tokens = query.lower().split()
    if len(tokens) > 20:
        return "qwen:7b"
    elif any(k in tokens for k in ["code", "program", "error", "bug", "syntax", "compile"]):
        return "phi3:latest"
    else:
        return "llama3:latest"

# --- Core Functions ---

def retrieve_chunks(query, top_k):
    """Retrieve top-k relevant chunks from Chroma."""
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    # Debug: show raw result keys and counts
    print("Raw result keys:", list(results.keys()))
    print(f"  documents: {len(results.get('documents', []))}")
    print(f"  metadatas: {len(results.get('metadatas', []))}")
    print(f"  distances: {len(results.get('distances', []))}")
    return results

def build_prompt(context, question):
    """Dynamically build the RAG prompt."""
    template = '''Use the following context to answer the user's question as accurately as possible.

=== Context ===
{context}

=== Question ===
{question}

=== Answer ===
'''
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
    return prompt.format(context=context, question=question)

def answer_question(query):
    """Full pipeline: retrieve -> filter with static threshold -> dynamic model -> call LLM -> return answer."""
    # 1) dynamic top_k
    top_k = dynamic_top_k(query)

    # 2) retrieve
    results = retrieve_chunks(query, top_k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    print(f"🎯 Retrieved {len(docs)} chunks")

    # 3) static threshold
    RELEVANCE_THRESHOLD = 1.0  # unchanged from original
    print(f"📏 Relevance threshold: {RELEVANCE_THRESHOLD:.3f}")
    relevant_docs = [
        (doc, meta) for doc, meta, dist in zip(docs, metas, distances) if dist <= RELEVANCE_THRESHOLD
    ]

    # 4) build prompt or fallback
    if not relevant_docs:
        print("⚠️ No relevant documents found based on similarity. Falling back to general LLM answer...")
        prompt_text = f"Answer based on your general knowledge: {query}"
    else:
        print("\n" + "="*60)
        print("📚 Relevant Retrieved Chunks")
        print("="*60)
        for idx, (doc, meta) in enumerate(relevant_docs, start=1):
            print(f"\n--- Chunk #{idx} ---")
            print(f"Content:\n{doc}")
            print(f"Metadata:\n{json.dumps(meta, indent=2)}")
            print("-" * 60)
        context = "\n\n".join(doc for doc, _ in relevant_docs)
        prompt_text = build_prompt(context, query)

    # 5) dynamic model selection
    model_name = dynamic_model_selection(query)
    print(f"\n🧠 Using model: {model_name}")
    llm = OllamaLLM(model=model_name, temperature=0.1)

    # 6) invoke LLM
    response = llm.invoke(prompt_text)
    return response.strip()

# --- Run ---

if __name__ == "__main__":
    try:
        user_query = input("\n📝 Please enter your query: ").strip()
        if not user_query:
            raise ValueError("Query cannot be empty. Please provide a valid query.")
        answer = answer_question(user_query)
        print("\n" + "="*60)
        print(f"\n✅ AI Response:\n{answer}\n")
    except Exception as e:
        print(f"Error: {e}")
    print("="*60 + "\n")
