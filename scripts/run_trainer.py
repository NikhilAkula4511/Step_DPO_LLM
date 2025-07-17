import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
import pdb


from trl import DPOConfig

# Add project root to PYTHONPATH to enable relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_train.step_wise_dpo_trainer import StepwiseDPOTrainer

from dataset.load_dataset import load_prm800k_dataset

# ----------------------
# Configurations
# ----------------------

model_name = "Qwen/Qwen2-0.5B-Instruct" # Replace with your base model
output_dir = "./dpo_outputs"

# ----------------------
# Load Tokenizer & Model
# ----------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Required for gpt2

model = AutoModelForCausalLM.from_pretrained(model_name)

# ----------------------
# Load Dataset
# ----------------------
jsonl_path =  r"C:\Users\91939\Nikhil_LLMs\Future_AGI\Data\prm800k\prm800k\data\phase1_test.jsonl"
dataset = load_prm800k_dataset(jsonl_path=jsonl_path ,tokenizer=tokenizer)
# dataset = dataset[0]

# dataset = Dataset.from_list(dataset)

# ----------------------
# Training Arguments
# ----------------------

training_args = DPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    output_dir="./outputs",
    remove_unused_columns=False,
    report_to=None,
    max_prompt_length=128,
    max_length=512,
    bf16=False,
    fp16=False,
    auto_find_batch_size=True,
)

# ----------------------
# DPO Trainer
# ----------------------

model = torch.compile(model)


trainer = StepwiseDPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# ----------------------
# Train
# ----------------------

if __name__ == "__main__":
    print(">>> About to start training")
    
    try:
        trainer.train()
    except Exception as e:
        import traceback
        traceback.print_exc()
    print(">>> Finished training")
    trainer.save_model(output_dir)  # Saves model + tokenizer
    tokenizer.save_pretrained(output_dir)
