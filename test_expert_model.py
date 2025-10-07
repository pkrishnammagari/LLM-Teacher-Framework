import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Base model and adapter
base_model_id = "microsoft/Phi-3-mini-4k-instruct"   # Official repo id casing [model card]
adapter_path = "phi-3-mini-rag-expert"

print("--- Loading base model and tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
# Ensure pad token is set for generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Choose dtype; "auto" lets Transformers pick from the checkpoint metadata
# On Apple Silicon or CPU, "eager" attention is the safe default
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    trust_remote_code=True,
    torch_dtype="auto",              # <- use torch_dtype instead of dtype
    device_map="auto",
    attn_implementation="eager",     # use "flash_attention_2" only if installed/supported
)

print(f"--- Loading adapter from: {adapter_path} ---")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

print("--- Merging adapter into the base model ---")
model = peft_model.merge_and_unload()  # returns the merged base HF model

# Align generation config
if getattr(model, "generation_config", None) is not None:
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

print("\n--- RAG Expert Model Ready! ---")
print("Ask a question about RAG systems. Type 'exit' to quit.")

while True:
    user_input = input("\n>>> You: ")
    if user_input.strip().lower() == "exit":
        break

    messages = [{"role": "user", "content": user_input}]

    # Create inputs using the model's chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    # Normalize to tensors and move to device
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(model.device)
    else:
        input_ids = inputs.to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)

    print(">>> RAG Expert: ", end="", flush=True)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = input_ids.shape[1]
    generated = outputs[0, prompt_len:]
    print(tokenizer.decode(generated, skip_special_tokens=True))
