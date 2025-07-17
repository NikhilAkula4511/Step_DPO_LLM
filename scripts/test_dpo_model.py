import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 1: Define the path to your saved model directory
model_path = "./dpo_outputs"  # Replace with your actual folder path

# Step 2: Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Set the prompt
prompt = "Question: How many legs do 3 dogs and 2 cats have together? Let's think step by step."

# Step 4: Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Step 5: Generate output
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Step 6: Decode the generated response
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 7: Print the output
print("Model Output:\n")
print(decoded_output)
