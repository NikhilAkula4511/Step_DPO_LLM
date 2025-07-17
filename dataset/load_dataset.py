import json
from datasets import Dataset

def load_custom_prm800k(jsonl_path):
    dataset = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt = entry["question"]["problem"]
                steps = entry.get("label", {}).get("steps", [])

                chosen_parts = []
                rejected_parts = []

                for step in steps:
                    completions = step.get("completions", [])
                    if not completions:
                        continue

                    rated_completions = [(c["text"], c.get("rating", 0)) for c in completions if "rating" in c]
                    positives = [text for text, rating in rated_completions if rating == 1]
                    negatives = [text for text, rating in rated_completions if rating == -1]

                    if positives:
                        chosen_parts.append(positives[0])
                    elif completions:
                        chosen_parts.append(completions[0]["text"])

                    if negatives:
                        rejected_parts.append(negatives[0])
                    elif completions:
                        rejected_parts.append(completions[-1]["text"])

                if chosen_parts and rejected_parts:
                    dataset.append({
                        "prompt": prompt,
                        "chosen": "\n".join(chosen_parts),
                        "rejected": "\n".join(rejected_parts)
                    })

            except json.JSONDecodeError as e:
                print("Skipping line due to JSON error:", e)

    return Dataset.from_list(dataset)


if __name__ == "__main__":
    jsonl_file = r"C:\Users\91939\Nikhil_LLMs\Future_AGI\Data\prm800k\prm800k\data\phase1_test.jsonl"
    dataset = load_custom_prm800k(jsonl_file)

    # Print the first example
    print(dataset[0])
