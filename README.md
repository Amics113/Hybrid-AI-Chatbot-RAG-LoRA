# 🚀 Hybrid AI Chatbot (RAG + LoRA + Llama3)

## 💡 Overview

This project is an intelligent AI chatbot that combines multiple AI techniques to provide accurate, contextual, and personalized responses.

It uses a **hybrid architecture**:

* 🔹 Llama3 (via Ollama) for general intelligence
* 🔹 LoRA fine-tuned model for personalized responses
* 🔹 RAG (Retrieval-Augmented Generation) for document-based answers
* 🔹 Memory system for maintaining conversation context

---

## 🧠 Key Features

* ✅ Hybrid AI routing (LoRA + Llama3)
* ✅ Personalized responses using fine-tuning (LoRA)
* ✅ Context-aware answers using RAG
* ✅ Chat memory for better conversations
* ✅ Local AI deployment (no API cost)
* ✅ Fault-tolerant backend (retry + fallback system)

---

## 🏗 Architecture

User → FastAPI Backend → Smart Routing
→ Personal Query → LoRA Model
→ General Query → Llama3 (Ollama)
→ Context Query → RAG Retrieval

---

## 🛠 Tech Stack

* **Backend:** FastAPI, Python
* **AI/ML:** Transformers, PEFT (LoRA), HuggingFace
* **LLM Runtime:** Ollama (Llama3)
* **Frontend:** HTML, CSS, JS
* **Database:** JSON / Vector Store (RAG)

---

## 📸 Demo

(Add screenshots here — very important)

---

## ⚙️ Setup Instructions

### 1. Clone repo

```bash
git clone <https://github.com/Amics113/Hybrid-AI-Chatbot-RAG-LoRA.git>
cd ai-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

```bash
ollama serve
ollama pull llama3:8b
```

### 4. Run backend

```bash
uvicorn app:app --reload
```

### 5. Open in browser

```
http://localhost:8000
```

---

## 🔥 Example Queries

* "Who created you?" → LoRA response
* "Explain HTML" → Llama3 response
* "Ask from documents" → RAG response

---

## 📈 Future Improvements

* Smarter AI-based routing (instead of keywords)
* Better fine-tuning dataset
* UI improvements
* Deployment on cloud

---

## 👨‍💻 Author

John – Computer Science Engineering Student
Interested in AI, Systems, and Scalable Solutions
