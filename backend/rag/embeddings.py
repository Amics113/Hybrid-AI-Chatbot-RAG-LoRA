import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"

def embed_text(text: str):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    return response.json()["embedding"]