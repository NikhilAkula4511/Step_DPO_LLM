📘 Dataset Description: Stepwise DPO from PRM800K
📁 Dataset Name
Stepwise Preference Dataset (Derived from PRM800K)

📚 Source
This dataset is adapted from the prm800k dataset, specifically designed for stepwise Direct Preference Optimization (DPO).

🧠 Purpose
This dataset is built for training stepwise reward models and preference-based fine-tuning of LLMs. The goal is to train models to prefer multi-step answers where each step improves upon the previous, encouraging better chain-of-thought reasoning.

🧾 Format
Each data point includes:

prompt: A natural language question requiring multi-step reasoning.

chosen: The preferred stepwise reasoning sequence.

rejected: The less preferred or incorrect reasoning sequence.

Example:

json
Copy code
{
  "prompt": "How many seconds are there in 3 days?",
  "chosen": "Step 1: There are 24 hours in a day.\nStep 2: So, 3 days = 3 × 24 = 72 hours.\nStep 3: There are 60 minutes in an hour.\nStep 4: 72 × 60 = 4320 minutes.\nStep 5: Each minute has 60 seconds, so 4320 × 60 = 259200 seconds.",
  "rejected": "Step 1: 3 days is 72 hours.\nStep 2: Maybe it’s 400000 seconds."
}
🗂 File Format
The raw data file is in JSONL (JSON Lines) format:

Each line is a JSON object with nested structure:

json
Copy code
{
  "question": {
    "problem": "string"
  },
  "label": {
    "steps": [
      {
        "completions": [
          {
            "text": "string",
            "rating": 1 or -1 or 0
          },
          ...
        ]
      }
    ]
  }
}
🔄 Preprocessing Logic
We use the following preprocessing steps (as implemented in load_dataset.py):

Parse each line of the .jsonl file.

Extract the problem as the prompt.

For each step:

Identify the best-rated (rating: 1) and worst-rated (rating: -1) completions.

Append the best to chosen and worst to rejected.

Concatenate stepwise completions into multi-line strings.

✅ Quality Checks
Entries without both positive and negative completions are skipped or handled gracefully.

Only examples with valid multi-step completions are retained.

📦 Output Dataset Format
We convert the final dataset to HuggingFace datasets.Dataset format with the following fields:

Field	Type	Description
prompt	string	The original multi-step question
chosen	string	Concatenated preferred steps
rejected	string	Concatenated rejected steps

🧪 Split
For experiments, we optionally support:

train.jsonl

val.jsonl

test.jsonl (e.g., phase1_test.jsonl)

🛠 Usage
python
Copy code
from datasets import load_dataset
dataset = load_dataset('path/to/your/dataset_script.py', data_dir='path/to/data')
👤 Author
Nikhil Akula

Dataset preparation script: load_dataset.py

Date: July 2025
