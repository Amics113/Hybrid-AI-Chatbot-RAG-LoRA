import os
import pickle
import numpy as np
from backend.rag.embeddings import embed_text

# ---------- PATHS ----------

INDEX_PATH = "data/vector_db/index.pkl"

TOP_K = 5
MAX_DISTANCE = 1.2


# ---------- LOAD VECTOR DB ----------

def _load_index():
    if not os.path.exists(INDEX_PATH):
        return None, []

    with open(INDEX_PATH, "rb") as f:
        data = pickle.load(f)

    # ✅ support tuple format from ingest.py
    if isinstance(data, tuple):
        return data[0], data[1]

    # ✅ support old dict format (future-safe)
    if isinstance(data, dict):
        return data["index"], data["texts"]

    return None, []
# ---------- RETRIEVE CONTEXT ----------

def retrieve_context(query: str) -> str:
    """
    Retrieve relevant document context using Ollama embeddings.
    """

    if index is None or not texts:
        return ""

    # ✅ Embed using Ollama (NOT HuggingFace)
    query_embedding = embed_text(query)
    query_vector = np.array([query_embedding]).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vector, TOP_K)

    results = []

    for dist, idx in zip(distances[0], indices[0]):

        if idx < 0:
            continue

        if dist > MAX_DISTANCE:
            continue

        results.append((dist, texts[idx]))

    if not results:
        return ""

    # Rerank (closest first)
    results.sort(key=lambda x: x[0])

    best_chunks = [text for _, text in results[:2]]

    return "\n\n".join(best_chunks)