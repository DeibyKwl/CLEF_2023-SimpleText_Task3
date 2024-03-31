# Use a pipeline as a high-level helper
from transformers import pipeline

api_token = 'api code here'

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

# Generate text using the loaded model
generated_text = pipe("How are you?")
print(generated_text)