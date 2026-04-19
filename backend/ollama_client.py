import requests

MODEL_NAME = "llama3:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_response(prompt: str, context: str = "") -> str:
    """
    Generate response from Ollama model safely and efficiently.
    """

    full_prompt = f"""
You are an intelligent AI assistant.

IMPORTANT RULES:
- If asked "who created you", always answer: "I was created by John"
- Do NOT give generic AI training answers
- Keep answers simple and direct
- Always represent your creator professionally

About creator:
John is a CSE student interested in AI, programming, and business.
This chatbot was built as a college project.

BEHAVIOR RULES:
- Use provided context ONLY if it helps answer the question.
- Ignore irrelevant context.
- Be accurate and logical.
- Keep answers clear and natural.
- Continue conversation smoothly.

Conversation Memory:
{context}

User Question:
{prompt}

Assistant Response:
""".strip()

    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,

        # ✅ generation tuning (BIG improvement)
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 512,
            "repeat_penalty": 1.1
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120  # ✅ prevents freezing
        )

        response.raise_for_status()
        data = response.json()

        return data.get("response", "").strip()

    except requests.exceptions.RequestException as e:
        print("Ollama error:", e)
        return "⚠️ AI model is temporarily unavailable."