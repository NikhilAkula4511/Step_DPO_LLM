from openai import OpenAI
import os
import time

# You can use Huggingface `transformers` + `AutoModelForCausalLM` instead if you have a local model
# This version assumes OpenAI GPT-4 (or any LLM with chat interface)

class StepwiseRewardModel:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def evaluate_stepwise(self, prompt, chosen, rejected):
        """
        Evaluate chosen vs rejected stepwise using GPT-4 as the judge.
        Returns a list of per-step preferences (1 for chosen better, 0 for rejected better, 0.5 for tie).
        """
        chosen_steps = self._split_steps(chosen)
        rejected_steps = self._split_steps(rejected)

        max_len = min(len(chosen_steps), len(rejected_steps))
        step_scores = []

        for i in range(max_len):
            comparison_prompt = f"""
You are an expert AI alignment judge.

Given a prompt and two step-by-step responses (Step {i+1} only), judge which one is better, more helpful, and correct.

Prompt: {prompt}

Chosen Step {i+1}: {chosen_steps[i]}
Rejected Step {i+1}: {rejected_steps[i]}

Who is better? Reply with "Chosen", "Rejected", or "Tie".
"""
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": comparison_prompt}],
                    temperature=0.0,
                )
                content = response.choices[0].message.content.strip().lower()
                if "chosen" in content:
                    step_scores.append(1)
                elif "rejected" in content:
                    step_scores.append(0)
                else:
                    step_scores.append(0.5)
            except Exception as e:
                print("Error during LLM evaluation:", e)
                step_scores.append(0.5)
            time.sleep(1)  # prevent rate limiting

        return step_scores

    def _split_steps(self, text):
        # Naive splitter - feel free to improve
        return [step.strip() for step in text.split('.') if step.strip()]

