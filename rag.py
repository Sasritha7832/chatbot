import os
import glob
import pickle
import numpy as np
import google.generativeai as genai
from PyPDF2 import PdfReader

# Directory configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'data', 'docs')
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'data', 'vector_store.pkl')

EMBEDDING_MODEL = 'models/text-embedding-004'

def chunk_text(text, chunk_size=1000, overlap=200):
    """Splits text into chunks of specified size and overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
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

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        msg = "GEMINI_API_KEY environment variable is not set on this server."
        print(f"ERROR: {msg}")
        return False, msg

    genai.configure(api_key=api_key)

    docs = []

    # Process TXT files
    for txt_file in glob.glob(os.path.join(DOCS_DIR, '*.txt')):
        with open(txt_file, 'r', encoding='utf-8') as f:
            docs.append({"filename": os.path.basename(txt_file), "text": f.read()})

    # Process PDF files
    for pdf_file in glob.glob(os.path.join(DOCS_DIR, '*.pdf')):
        text = extract_text_from_pdf(pdf_file)
        if text:
            docs.append({"filename": os.path.basename(pdf_file), "text": text})

    if not docs:
        print("No documents found in data/docs/. Vector store is empty.")
        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump({"chunks": [], "embeddings": []}, f)
        return True, None

    all_chunks = []

    print(f"Processing {len(docs)} document(s)...")
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(f"Source: {doc['filename']}\n\n{chunk}")

    # Generate embeddings in batches of 100 to avoid API limits
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    try:
        all_embeddings = []
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            all_embeddings.extend(result['embedding'])

        with open(VECTOR_STORE_PATH, 'wb') as f:
            pickle.dump({
                "chunks": all_chunks,
                "embeddings": np.array(all_embeddings)
            }, f)

        print(f"Successfully saved {len(all_chunks)} embeddings to vector store.")
        return True, None
    except Exception as e:
        msg = str(e)
        print(f"Error generating embeddings: {msg}")
        return False, msg

def cosine_similarity(a, b):
    # Avoid division by zero
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def retrieve_context(query, top_k=3):
    """Retrieves the top_k most similar chunks for a given query."""
    if not os.path.exists(VECTOR_STORE_PATH):
        return ""
        
    with open(VECTOR_STORE_PATH, 'rb') as f:
        store = pickle.load(f)
        
    chunks = store.get("chunks", [])
    embeddings = store.get("embeddings", [])
    
    if len(chunks) == 0 or len(embeddings) == 0:
        return ""
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""
        
    genai.configure(api_key=api_key)
    
    try:
        # Embed the query
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(result['embedding'])
        
        # Calculate similarities
        similarities = [cosine_similarity(query_embedding, emb) for emb in embeddings]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Only keep indices with a reasonable similarity score (e.g. > 0.55)
        filtered_indices = [i for i in top_indices if similarities[i] > 0.55]
        
        retrieved_chunks = [chunks[i] for i in filtered_indices]
        
        if not retrieved_chunks:
            return ""
            
        return "\n\n---\n\n".join(retrieved_chunks)
    except Exception as e:
        print(f"Retrieval error: {e}")
        return ""

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    ingest_documents()
