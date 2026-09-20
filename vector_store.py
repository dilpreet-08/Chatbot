import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from data_loader import load_documents

INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "documents.pkl"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load or generate index and documents
if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
    print("Loading cached vector index and documents...")
    index = faiss.read_index(INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        documents = pickle.load(f)
else:
    print("Building vector index from documents in knowledge_base (this may take a minute)...")
    documents = load_documents()
    embeddings = model.encode(documents, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    try:
        faiss.write_index(index, INDEX_FILE)
        with open(DOCS_FILE, "wb") as f:
            pickle.dump(documents, f)
        print("Vector index and documents cached successfully.")
    except Exception as e:
        print(f"Warning: could not cache index: {e}")

def search(query, top_k=3):
    # Encode query
    query_vector = model.encode([query], normalize_embeddings=True)
    query_vector = np.array(query_vector).astype("float32")

    # Search
    scores, indices = index.search(query_vector, top_k)

    results = []
    if len(scores) == 0 or len(scores[0]) == 0:
        return results, 0.0

    best_score = float(scores[0][0])

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(documents):
            if score > best_score * 0.80:   # only keep chunks close to best
                results.append((documents[idx], float(score)))

    return results, best_score

