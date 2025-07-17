# Step_DPO_LLM
# Stepwise DPO Implementation – Future_AGI

This repository contains the implementation of the **Stepwise DPO** methodology inspired by the paper [Let's Verify Step by Step (OpenAI, 2024)](https://arxiv.org/html/2408.15240v1).  
The goal is to use **LLM-based stepwise reward modeling** instead of human annotators to train preference models with more interpretability and granularity.

---

## 📁 Project Structure

Future_AGI/
├── scripts/
│ └── run_trainer.py # Script to launch training using StepwiseDPOTrainer
├── src/
│ ├── trainers/
│ │ └── stepwise_dpo_trainer.py # Custom StepwiseDPOTrainer logic (extends HF DPOTrainer)
│ └── reward_model/
│ └── reward_model.py # Step-level reward scoring using LLM or scoring function
└── checkpoints/
└── stepwise_dpo/ # Stores trained models and checkpoints

yaml
Copy
Edit

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/Future_AGI.git
cd Future_AGI
2. Create Environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # or use .venv\Scripts\activate on Windows
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Download Data (e.g., prm800k)
You can download or preprocess the dataset manually. By default, our loader expects the prm800k format with stepwise reasoning and preference pairs.

🧠 Core Components
✅ Stepwise Reward Model (reward_model.py)
Uses an LLM (e.g., GPT-4, Claude) to score each step of reasoning.

Evaluates whether each reasoning step in a response is correct or preferred.

Returns per-step scores used to compute final aggregated reward.

⚙️ Stepwise DPO Trainer (stepwise_dpo_trainer.py)
Subclass of Hugging Face's DPOTrainer.

Accepts step-level reward signals.

Aggregates stepwise scores into a final reward using arithmetic or weighted average.

Trains the policy model with preference pairs and stepwise supervision.

📜 Run Script (run_trainer.py)
Entry point to train using StepwiseDPOTrainer.

Pass paths, model configs, and training parameters via CLI args or config dict.

🧪 Running Training
bash
Copy
Edit
python scripts/run_trainer.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --dataset_path "./data/prm800k_subset.json" \
    --output_dir "./checkpoints/stepwise_dpo" \
    --max_steps 500 \
    --per_device_train_batch_size 4 \
    --reward_model "gpt-4" \
    --evaluation_strategy "steps"
Customize as needed.

📊 Evaluation
After training, you can evaluate the model by sending a prompt and checking generated output scores from reward_model.py.

python
Copy
Edit
from src.reward_model.reward_model import StepwiseRewardModel

rewarder = StepwiseRewardModel(model_name="gpt-4")
output_score = rewarder.evaluate(prompt, generation)
print("Stepwise Score:", output_score)
📈 Results (Example)
Metric	Value
Training Steps	500
Eval Metric	Reward Score ↑
Model Used	LLaMA-2 7B (LoRA)
Reward Source	GPT-4 (OpenAI API)

✅ Next Steps
 Improve reward aggregation (e.g., position-based weighting)

 Evaluate generalization on other stepwise datasets

 Add few-shot prompting for better reward quality

 Bonus: Try with mistral-7b or phi-2 for sub-1.3B experiments

🧾 AI Usage Log
See LLM_USAGE.md for where external LLMs were queried to assist with:

Generating scoring prompts

Designing trainer subclasses

Formatting synthetic pairs (if needed)

📦 Reproducibility
bash
Copy
Edit
# Create from scratch
conda env create -f environment.yml
# OR
pip install -r requirements.txt
Ensure OpenAI or Anthropic API keys are set in your environment if using external APIs.

👨‍🔬 References
Let’s Verify Step by Step (OpenAI)

Baseline GitHub: dvlab-research/Step-DPO

Hugging Face DPOTrainer
