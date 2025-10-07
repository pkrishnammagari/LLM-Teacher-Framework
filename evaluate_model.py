import requests
import json
import time
import statistics
from tqdm import tqdm

# --- 1. Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "command-r"  # Our "Teacher" LLM
RESULTS_FILE = "results.jsonl"

# --- 2. The Judge's System Prompt ---
# This is the most critical part. We are instructing the LLM on how to be a fair judge.
JUDGE_SYSTEM_PROMPT = """
You are an impartial and expert AI assistant. Your task is to evaluate the quality of a student model's answer based on a given instruction and a reference (ground-truth) answer.

Your evaluation should consider the following criteria:
1.  **Relevance and Correctness:** Is the student's answer factually correct and relevant to the instruction?
2.  **Completeness:** Does the answer address all parts of the instruction?
3.  **Clarity:** Is the answer clear, concise, and easy to understand?

You must provide a score from 1 to 10, where:
- 1 means the answer is completely wrong or irrelevant.
- 5 means the answer is partially correct but has significant flaws.
- 10 means the answer is perfect, accurate, and comprehensive.

You MUST provide your output in a strict JSON format with no other text. The JSON object must contain two keys:
1.  `"score"`: An integer from 1 to 10.
2.  `"justification"`: A brief, one-sentence explanation for your score.

Example format:
{"score": 8, "justification": "The answer is factually correct and covers the main points, but could be slightly more concise."}
"""

# --- 3. Load the Results to be Evaluated ---
print(f"--- Loading results from {RESULTS_FILE} ---")
with open(RESULTS_FILE, 'r') as f:
    results_data = [json.loads(line) for line in f]
print(f"Loaded {len(results_data)} results to evaluate.")

# --- 4. Evaluate Each Result ---
scores = []
with open("evaluation_details.log", "w") as log_file:
    for item in tqdm(results_data, desc="Evaluating Answers"):
        instruction = item['instruction']
        ground_truth = item['ground_truth_output']
        model_answer = item['model_generated_output']

        # Construct the full prompt for the judge
        eval_prompt = f"""
        **Instruction:**
        {instruction}

        **Reference Answer:**
        {ground_truth}

        **Student Model's Answer:**
        {model_answer}
        """

        try:
            payload = {
                "model": JUDGE_MODEL,
                "prompt": eval_prompt,
                "system": JUDGE_SYSTEM_PROMPT,
                "stream": False,
                "options": { "temperature": 0.0 } # Low temperature for consistent, objective scoring
            }

            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            
            response_text = response.json()['response']
            judge_feedback = json.loads(response_text)
            
            score = judge_feedback.get("score")
            justification = judge_feedback.get("justification", "")

            if isinstance(score, int) and 1 <= score <= 10:
                scores.append(score)
                log_entry = f"Instruction: {instruction}\nScore: {score}\nJustification: {justification}\n---\n"
                log_file.write(log_entry)
            else:
                log_file.write(f"Instruction: {instruction}\nError: Invalid score received: {score}\n---\n")

        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            error_entry = f"Instruction: {instruction}\nError during evaluation: {e}\n---\n"
            log_file.write(error_entry)
            time.sleep(2)

# --- 5. Calculate and Print Final Score ---
if scores:
    average_score = statistics.mean(scores)
    print("\n--- Evaluation Complete ---")
    print(f"Total answers evaluated: {len(scores)}/{len(results_data)}")
    print(f"**Average Score: {average_score:.2f} / 10**")
else:
    print("\n--- Evaluation Complete ---")
    print("No valid scores were recorded.")