import pandas as pd
import torch
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
#     print("cuda")
# else:
#     torch.set_default_device("cpu")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def get_tsv_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['snt_id'].tolist(), data['source_snt'].tolist(), data['query_text'].tolist()

def simplify_text(sentences):
    simplified_texts = []
    
    for sentence in sentences:
        print(sentence)
        
        input_text = f'''
            <s>
            [INST]
            Do not oversimplify it. Make sure it is shorter, but not too short. Omit the source link website, do not include them. Simplify the following sentence: {sentence}
            [/INST] This sentence can be simplified as follows:
            '''
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate simplified text
        attention_mask = torch.ones_like(input_ids)
        pad_token_id = tokenizer.eos_token_id

        output_ids = model.generate(input_ids, max_length=400, attention_mask=attention_mask, pad_token_id=pad_token_id)
        simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        index = simplified_text.find('[/INST] This sentence can be simplified as follows:')
        simplified_text = simplified_text[index + len('[/INST] This sentence can be simplified as follows:'):].strip().strip('"')
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

    results_df.to_csv("Task3_TopGap_mistral.tsv", sep='\t', index=False)

def main():
    if len(sys.argv) != 2:
        print('Usage: python Task3_mistralai.py <filename1>')
        return
    file_path = sys.argv[1]
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)

    simplified_texts = simplify_text(source_sentences)

    create_tsv(snt_ids, simplified_texts)

if __name__ == '__main__':
    main()