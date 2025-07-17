import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.reward_model import StepwiseRewardModel


api_key = api_key# Make sure it's exported
model = StepwiseRewardModel(api_key)

prompt = "Why do humans need sleep?"
chosen = "Sleep helps the brain consolidate memories. It also supports muscle repair. It regulates metabolism."
rejected = "Humans sleep because they are tired. Sleeping is fun. Everyone sleeps."

scores = model.evaluate_stepwise(prompt, chosen, rejected)
print("Stepwise Judgments:", scores)
