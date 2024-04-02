import pandas as pd
import torch

###################################################################################
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

###################################################################################


def get_tsv_data(file_path):

    snt_ids = []
    source_sentences = []
    query_texts = []

    data = pd.read_csv(file_path, sep='\t')
    
    for index, row in data.iterrows():
        query_text = row['query_text']
        snt_id = row['snt_id']
        source_snt = row['source_snt']

        query_texts.append(query_text)
        snt_ids.append(snt_id)
        source_sentences.append(source_snt)

    return snt_ids, source_sentences, query_texts

# f"[INST]<<SYS>>You are a college student<<SYS>>\n Simplify the following sentence: \'{sentence}\'[INST]" 
def simplify_text(sentences):
    simplified_texts = []

    for sentence in sentences:
        # Preprocess the input sentence
        input_text = f'''
            <s>
            [INST]
            Simplify the following sentence
            Sentence: {sentence}
            [/INST]
            </s>
            '''
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate simplified text
        output_ids = model.generate(input_ids, max_length=1000)
        simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)


        simplified_texts.append(simplified_text)

    return simplified_texts

def create_tsv(snt_ids, summaries):
    results_df = pd.DataFrame({
        "run_id": ["TopGap_task3_run1"] * len(snt_ids),
        "manual": 0,
        "snt_id": snt_ids,
        "simplified_snt": summaries
    })

    # Write DataFrame to TSV file
    results_df.to_csv("Task3_TopGap.tsv", sep='\t', index=False)


def main():
    file_path = 'simpletext_task3_train_2.tsv'
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)

    # Simplify the query texts
    simplified_texts = simplify_text(source_sentences)

    create_tsv(snt_ids, simplified_texts)

if __name__ == '__main__':
    main()