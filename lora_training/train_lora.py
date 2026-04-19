import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_name = "D:/ai/models/tinyllama"

# Tokenizer (LOCAL ONLY)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=False,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token

# Model (NO 4-bit — stable version)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    local_files_only=True
)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Dataset
dataset = load_dataset("json", data_files="dataset.json")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize)

# Training
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    output_dir="./lora_output",
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)

trainer.train()

model.save_pretrained("lora_adapter")