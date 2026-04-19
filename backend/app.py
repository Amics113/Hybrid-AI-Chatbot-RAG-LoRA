from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 🔥 HYBRID IMPORTS
from backend.lora_client import generate_lora_response

from backend.auth.auth import authenticate_user, register_user
from backend.ollama_client import generate_response
from backend.rag.retriever import retrieve_context
from backend.memory.chat_memory import add_message, get_recent_memory


# 🔥 ROUTER FUNCTION
def is_personal_query(prompt: str):
    keywords = [
        "who created you",
        "your creator",
        "about you",
        "who made you",
        "your project",
        "who is john",
        "about yourself"
    ]

    prompt = prompt.lower()
    return any(k in prompt for k in keywords)


app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ---------- PAGES ----------

@app.get("/", response_class=HTMLResponse)
def login_page():
    return open("frontend/login.html", encoding="utf-8").read()


@app.get("/register", response_class=HTMLResponse)
def register_page():
    return open("frontend/register.html", encoding="utf-8").read()


@app.get("/chat", response_class=HTMLResponse)
def chat_page():
    return open("frontend/index.html", encoding="utf-8").read()


# ---------- AUTH ----------

@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    register_user(username, password)
    return RedirectResponse("/", status_code=302)


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if authenticate_user(username, password):
        return {"status": "ok"}

    raise HTTPException(status_code=401, detail="Invalid credentials")


# ---------- CHAT API ----------

class ChatRequest(BaseModel):
    username: str
    prompt: str


@app.post("/chat")
def chat(req: ChatRequest):

    # 🔹 Save user message
    add_message(req.username, "User", req.prompt)

    # 🔹 Memory
    memory = get_recent_memory(req.username)

    # 🔹 RAG context
    if len(req.prompt.split()) < 6:
        context = ""
    else:
        context = retrieve_context(req.prompt)

    combined = f"{context}\n{memory}".strip()

    # 🔥 HYBRID LOGIC (MAIN CHANGE)
    if is_personal_query(req.prompt):
        answer = generate_lora_response(req.prompt)
    else:
        answer = generate_response(req.prompt, combined)

    # 🔹 Save bot reply
    add_message(req.username, "Bot", answer)

    return {"response": answer}