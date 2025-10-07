import json
from sklearn.model_selection import train_test_split
import random

# --- Configuration ---
INPUT_FILE = "rag_dataset.jsonl"
TRAIN_FILE = "train_dataset.jsonl"
TEST_FILE = "test_dataset.jsonl"
TEST_SIZE = 0.2  # 20% of the data will be used for testing
RANDOM_SEED = 42 # Ensures the split is the same every time we run it

print(f"--- Loading data from {INPUT_FILE} ---")
with open(INPUT_FILE, 'r') as f:
    data = [json.loads(line) for line in f]

print(f"Loaded {len(data)} total examples.")

# --- Split the Data ---
# We use a fixed random_state to ensure the split is reproducible.
train_data, test_data = train_test_split(
    data, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED
)

print(f"Splitting data: {len(train_data)} for training, {len(test_data)} for testing.")

# --- Save the Split Files ---
def save_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

save_to_jsonl(train_data, TRAIN_FILE)
print(f"--- Saved training data to {TRAIN_FILE} ---")

save_to_jsonl(test_data, TEST_FILE)
print(f"--- Saved test data to {TEST_FILE} ---")