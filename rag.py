import os
import glob
import pickle
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

load_dotenv()

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'data', 'docs')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'data', 'vector_store.pkl')

# Lazy-loaded local embedding model (no API key needed)
_embedding_model = None


def get_embedding_model():
    """Loads the local sentence-transformer model (cached after first load)."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model (first-time download ~90MB)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded.")
    return _embedding_model


def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


def ingest_documents():
    """
    Reads all docs, chunks them, embeds them, and saves to vector store.
    Returns (True, None) on success or (False, error_message) on failure.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    docs = []

    for txt_file in glob.glob(os.path.join(DOCS_DIR, '*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            docs.append({"filename": os.path.basename(txt_file), "text": f.read()})

    for pdf_file in glob.glob(os.path.join(DOCS_DIR, '*.pdf')):
        text = extract_text_from_pdf(pdf_file)
        if text:
            docs.append({"filename": os.path.basename(pdf_file), "text": text})

    if not docs:
        print("No documents found. Vector store is empty.")
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump({"chunks": [], "embeddings": []}, f)
        return True, None

    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append(f"Source: {doc['filename']}\n\n{chunk}")

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    try:
        model = get_embedding_model()
        all_embeddings = model.encode(all_chunks, show_progress_bar=False)

        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump({
                "chunks": all_chunks,
                "embeddings": np.array(all_embeddings)
            }, f)

        print(f"Saved {len(all_chunks)} embeddings to vector store.")
        return True, None

    except Exception as e:
        msg = str(e)
        print(f"Embedding error: {msg}")
        return False, msg


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def retrieve_context(query, top_k=3):
    """Retrieves the top_k most relevant chunks for a query."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return ""

    with open(VECTOR_STORE_PATH, 'rb') as f:
        store = pickle.load(f)

    chunks = store.get("chunks", [])
    embeddings = store.get("embeddings", [])

    if len(chunks) == 0 or len(embeddings) == 0:
        return ""

    try:
        model = get_embedding_model()
        query_embedding = model.encode([query])[0]

        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        filtered = [i for i in top_indices if similarities[i] > 0.40]
        retrieved = [chunks[i] for i in filtered]

        return "\n\n---\n\n".join(retrieved) if retrieved else ""

    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""


if __name__ == '__main__':
    ingest_documents()
