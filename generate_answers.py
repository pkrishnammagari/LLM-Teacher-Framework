import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm # For a nice progress bar

# --- 1. Configuration ---
base_model_id = "microsoft/phi-3-mini-4k-instruct"
adapter_path = "phi-3-mini-rag-expert"
test_dataset_file = "test_dataset.jsonl"
results_file = "results.jsonl"

# --- 2. Load the Fine-Tuned Model and Tokenizer ---
print("--- Loading base model, tokenizer, and adapter ---")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",
)

# Merge the adapter into the base model
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
print("--- Model loaded and merged successfully ---")

# --- 3. Load the Test Dataset ---
print(f"--- Loading test data from {test_dataset_file} ---")
test_dataset = load_dataset('json', data_files=test_dataset_file, split='train')

# --- 4. Generate Answers ---
results = []
# Use tqdm for a progress bar
for item in tqdm(test_dataset, desc="Generating Answers"):
    instruction = item['instruction']
    ground_truth = item['output']

    # Format the input using the model's chat template
    messages = [{"role": "user", "content": instruction}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    attention_mask = torch.ones_like(inputs)

    # Generate a response
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=512, # Allow for detailed answers
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode the generated part of the response
    input_length = inputs.shape[1]
    response = outputs[0][input_length:]
    generated_answer = tokenizer.decode(response, skip_special_tokens=True)

    # Store the question, the original answer, and the model's new answer
    results.append({
        "instruction": instruction,
        "ground_truth_output": ground_truth,
        "model_generated_output": generated_answer
    })

# --- 5. Save the Results ---
with open(results_file, 'w') as f:
    for result in results:
        f.write(json.dumps(result) + '\n')

print(f"\n--- Generation complete. Results saved to {results_file} ---")