import pandas as pd
import torch
import sys
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
#     print("cuda")
# else:
#     torch.set_default_device("cpu")

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

def get_tsv_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['snt_id'].tolist(), data['source_snt'].tolist(), data['query_text'].tolist()

def simplify_text(sentences):
    simplified_texts = []
    
    for sentence in sentences:
        print(sentence)
        pad_token_id = tokenizer.eos_token_id

        prompt = f'''<|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            You know how to make a sentence easier to read and understand.
            Whenever you simplify make sure to output only one sentence.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Simplify the following sentence: {sentence}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
            Here is a simplified version of the sentence:
            '''

        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = pipeline(prompt, max_new_tokens=10000, eos_token_id=terminators, pad_token_id=pad_token_id)
        simplified_text = outputs[0]["generated_text"]

        index = simplified_text.find('Here is a simplified version of the sentence:')
        simplified_text = simplified_text[index + len('Here is a simplified version of the sentence:'):].strip().strip('"')
        print(simplified_text+'\n')

        simplified_texts.append(simplified_text)

    return simplified_texts

def create_tsv(snt_ids, summaries):
    results_df = pd.DataFrame({
        "run_id": ["TopGap_task3_run1"] * len(snt_ids),
        "manual": 0,
        "snt_id": snt_ids,
        "simplified_snt": summaries
    })

    results_df.to_csv("Task3_TopGap_Llama3.tsv", sep='\t', index=False)

def main():
    if len(sys.argv) != 2:
        print('Usage: python Task3_Llama3.py <filename1>')
        return
    file_path = sys.argv[1]
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)
    simplified_texts = simplify_text(source_sentences)
    create_tsv(snt_ids, simplified_texts)

if __name__ == '__main__':
    main()