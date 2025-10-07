import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- 1. Configuration ---
model_id = "microsoft/phi-3-mini-4k-instruct"
dataset_file = "rag_dataset.jsonl"
new_model_name = "phi-3-mini-rag-expert"

# --- 2. Load the Dataset ---
print("--- Loading dataset ---")
dataset = load_dataset("json", data_files=dataset_file, split="train")

# Pre-format into a single 'text' column
def to_text(example):
    example["text"] = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return example

keep_cols = ["text"]
drop_cols = [c for c in dataset.column_names if c not in keep_cols]
dataset = dataset.map(to_text, remove_columns=drop_cols)

# --- 3. Load the Tokenizer ---
print("--- Loading tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- 4. Load the Model ---
print("--- Loading model ---")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,      # <- use torch_dtype
    device_map="auto",
    attn_implementation="eager"      # avoid flash-attn on macOS
)

# --- 5. Configure LoRA ---
print("--- Configuring LoRA ---")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# --- 6. TrainingArguments ---
print("--- Configuring training arguments ---")
training_args = TrainingArguments(
    output_dir=new_model_name,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    logging_steps=20,
    learning_rate=2e-4,
    bf16=True,
    dataloader_num_workers=0,
    save_strategy="epoch",
    report_to="none",
)

# --- 7. Trainer ---
# Use tokenizer=tokenizer (not processing_class) for this TRL version
# Avoid deprecated dataset_text_field/max_seq_length by providing a formatting_func
print("--- Initializing trainer ---")
def formatting_prompts_func(batch):
    return batch["text"]

trainer = SFTTrainer(
    model=model,                      # already PEFT-wrapped; do NOT pass peft_config here
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,              # <- use 'tokenizer' for this TRL version
    packing=False,
    formatting_func=formatting_prompts_func,
)

# --- 8. Train ---
print("--- Starting training ---")
trainer.train()

# --- 9. Save ---
print("--- Saving trained model adapter ---")
trainer.save_model(new_model_name)

print(f"--- Training complete! Model adapter saved to '{new_model_name}' ---")
