import json
import os
from typing import Dict, List

MEMORY_FILE = "backend/memory/chat_history.json"
MAX_MESSAGES_PER_USER = 100   # ✅ prevents infinite growth


# ---------- INTERNAL HELPERS ----------

def _load() -> Dict[str, List[dict]]:
    """Load memory safely."""
    if not os.path.exists(MEMORY_FILE):
        return {}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # ✅ prevents crash if file corrupts
        return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------- MEMORY API ----------

def add_message(username: str, role: str, message: str):
    data = _load()

    if username not in data:
        data[username] = []

    data[username].append({
        "role": role,
        "message": message
    })

    # ✅ auto memory trimming (VERY IMPORTANT)
    data[username] = data[username][-MAX_MESSAGES_PER_USER:]

    _save(data)


def get_memory(username: str) -> str:
    data = _load()

    if username not in data:
        return ""

    return "\n".join(
        f"{m['role']}: {m['message']}"
        for m in data[username]
    )


# ---------- SMART MEMORY (USED BY AI) ----------

def get_recent_memory(username: str, limit: int = 6) -> str:
    data = _load()

    if username not in data:
        return ""

    recent = data[username][-limit:]

    return "\n".join(
        f"{m['role']}: {m['message']}"
        for m in recent
    )