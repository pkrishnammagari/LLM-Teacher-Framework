import requests
import json
import time
import os

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "command-r"
DATASET_FILENAME = "rag_dataset.jsonl"
NUM_EXAMPLES_TO_GENERATE = 500

# --- The Final, Strictest RAG System Prompt ---
SYSTEM_PROMPT = """
You are a world-class expert and senior AI engineer specializing in Retrieval-Augmented Generation (RAG) systems. Your task is to generate a diverse set of high-quality questions and detailed answers for a fine-tuning dataset.

**Absolute Rules:**
1.  **VARY YOUR QUESTIONS.** Create a mix of questions. Some should use the full phrase "Retrieval-Augmented Generation", but MANY should use the common acronym "RAG".
2.  The topics must cover the entire RAG pipeline: embeddings, vector databases, chunking, etc.
3.  **YOUR RESPONSE MUST BE A SINGLE, COMPLETE, RAW JSON OBJECT.**
4.  The JSON object MUST contain exactly two keys: `"instruction"` and `"output"`.
5.  **CRITICAL: Ensure the final output is a perfectly valid JSON object with no syntax errors, such as missing commas.**

**Example of PERFECT output format:**
{"instruction": "What is RAG?", "output": "RAG stands for Retrieval-Augmented Generation. It's a technique that enhances large language models by allowing them to retrieve relevant information from an external knowledge base before generating a response."}

Now, generate a new, unique, and syntactically perfect JSON object following all rules.
"""

# --- Resumable Script Logic ---
existing_examples = 0
if os.path.exists(DATASET_FILENAME):
    with open(DATASET_FILENAME, 'r') as f:
        existing_examples = sum(1 for line in f)

remaining_examples = NUM_EXAMPLES_TO_GENERATE - existing_examples

print(f"--- Starting RAG Dataset Generation using '{MODEL_NAME}' ---")
print(f"Found {existing_examples} existing examples in '{DATASET_FILENAME}'.")
if remaining_examples <= 0:
    print("Dataset already has the desired number of examples. Exiting.")
else:
    print(f"Generating {remaining_examples} more examples...")

    with open(DATASET_FILENAME, 'a') as f:
        for i in range(remaining_examples):
            print(f"Generating example {existing_examples + i + 1}/{NUM_EXAMPLES_TO_GENERATE}...")
            try:
                payload = {
                    "model": MODEL_NAME,
                    "prompt": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.8, "num_predict": 8192}
                }
                
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) 
                response.raise_for_status()
                response_text = response.json()['response'].strip()
                
                try:
                    data = json.loads(response_text, strict=False)
                except json.JSONDecodeError:
                    print("  - JSON decode failed. Attempting to clean markdown...")
                    start_index = response_text.find('{')
                    end_index = response_text.rfind('}')
                    
                    if start_index != -1 and end_index != -1 and end_index > start_index:
                        json_text = response_text[start_index:end_index+1]
                        data = json.loads(json_text, strict=False)
                        print("  - Cleaning successful!")
                    else:
                        raise ValueError("Could not find valid JSON object in the string.")

                if "instruction" in data and "output" in data:
                    f.write(json.dumps(data) + '\n')
                else:
                    print(f"  - Discarding: Cleaned JSON missing keys. Response: {response_text}")

            except (requests.exceptions.RequestException, json.JSONDecodeError, ValueError) as e:
                print(f"  - Discarding: An unrecoverable error occurred: {e}")
                time.sleep(5)

print(f"\n--- Dataset generation complete. Data saved to '{DATASET_FILENAME}' ---")