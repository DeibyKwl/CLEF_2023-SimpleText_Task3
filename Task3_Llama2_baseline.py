import pandas as pd
import torch
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

# if torch.cuda.is_available():
#     torch.set_default_device("cuda")
#     print("cuda")
# else:
#     torch.set_default_device("cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

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
            <<SYS>>
            Only output the result, 
            do not explain your method, 
            do not say that you simplify something,
            do not explain the sentence,
            do not start the output with a prompt and colon,
            make sure the output has sentences with words,
            Remember to not include any quotation marks in your output
            <</SYS>>
            Simplify the following sentence: {sentence}
            [/INST]
            '''
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate simplified text
        output_ids = model.generate(input_ids, max_length=1000)
        simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        index = simplified_text.find('[/INST]')
        simplified_text = simplified_text[index + len('[/INST]'):].strip()
        #simplified_text = simplified_text[:-1]
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

    results_df.to_csv("Task3_TopGap_Llama2.tsv", sep='\t', index=False)


def main():
    if len(sys.argv) != 2:
        print('Usage: python Task3_Llama2_baseline.py <filename>')
        return
    
    file_path = sys.argv[1]
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)
    simplified_texts = simplify_text(source_sentences)
    create_tsv(snt_ids, simplified_texts)

if __name__ == '__main__':
    main()