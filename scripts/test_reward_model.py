import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.reward_model import StepwiseRewardModel


api_key = "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"  # Replace with your OpenAI API key
  # Make sure it's exported
model = StepwiseRewardModel(api_key)

prompt = "Why do humans need sleep?"
chosen = "Sleep helps the brain consolidate memories. It also supports muscle repair. It regulates metabolism."
rejected = "Humans sleep because they are tired. Sleeping is fun. Everyone sleeps."

scores = model.evaluate_stepwise(prompt, chosen, rejected)
print("Stepwise Judgments:", scores)
