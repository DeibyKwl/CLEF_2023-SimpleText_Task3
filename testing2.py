import torch
import transformers

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")


from transformers import AutoTokenizer

model = "meta-llama/Llama-2-7b-chat-hf"
# model = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float32,
    device_map="auto",
    max_new_tokens=7500,
)

sequences = pipeline(
    "Where is Paris?",
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
# print(len(sequences))
for seq in sequences:
    print(seq)
    result = seq['generated_text']