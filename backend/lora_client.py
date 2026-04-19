from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

MODEL_PATH = "D:/ai/models/tinyllama"
LORA_PATH = "D:/ai/ai-chatbot/lora_training/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.to(device)
model.eval()


def generate_lora_response(prompt: str) -> str:
    formatted_prompt = f"""
You are an AI created by John. Answer accordingly.

### Instruction: {prompt}
### Response:
"""

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.1,   # 🔥 VERY LOW = less hallucination
            do_sample=False    # 🔥 deterministic
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return decoded.split("### Response:")[-1].strip()
