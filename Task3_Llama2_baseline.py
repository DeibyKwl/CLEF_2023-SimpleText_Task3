import pandas as pd
import torch

import sys
sys.path.append('../easse')

from easse.fkgl import corpus_fkgl
from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    torch.set_default_device("cuda")
    print("cuda")
else:
    torch.set_default_device("cpu")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


def get_tsv_data(file_path):

    snt_ids = []
    source_sentences = []
    query_texts = []

    data = pd.read_csv(file_path, sep='\t')
    
    for _, row in data.iterrows():
        query_text = row['query_text']
        snt_id = row['snt_id']
        source_snt = row['source_snt']

        query_texts.append(query_text)
        snt_ids.append(snt_id)
        source_sentences.append(source_snt)

    return snt_ids, source_sentences, query_texts


def simplify_text(sentences):
    simplified_texts = []

    for sentence in sentences:
        # Preprocess the input sentence
        input_text = f'''
            <s>
            [INST]
            <<SYS>>
            Only output the result, 
            do not explain your method, 
            do not say that you simplify something, 
            do not talk about you classifier, 
            do not explain the sentence, 
            do not start the output with a prompt and colon,
            and please make sure the output has sentences with words
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
        simplified_text = simplified_text[:-1]
        simplified_texts.append(simplified_text)

    return simplified_texts

def create_tsv(snt_ids, summaries):
    results_df = pd.DataFrame({
        "run_id": ["TopGap_task3_run1"] * len(snt_ids),
        "manual": 0,
        "snt_id": snt_ids,
        "simplified_snt": summaries
    })

    results_df.to_csv("Task3_TopGap.tsv", sep='\t', index=False)

def get_simplified_tsv_data():
    simplified_snts = []
    
    data = pd.read_csv('Task3_TopGap.tsv', sep='\t')
    
    for _, row in data.iterrows():
        simplified_snt = row['simplified_snt']
        simplified_snts.append(simplified_snt)
    
    return simplified_snts

def create_tsv_metrics(snt_ids, source_sentences, simplified_snts):

    fkgl_source = []
    fkgl_simplify = []
    compression_ratios = []
    deletions_proportions = []

    for source_sentence, simplify_sentence in zip(source_sentences, simplified_snts):
        fkgl_sentence1 = round(corpus_fkgl([source_sentence]), 3)
        fkgl_sentence2 = round(corpus_fkgl([simplify_sentence]), 3)
        fkgl_source.append(fkgl_sentence1)
        fkgl_simplify.append(fkgl_sentence2)

        compression_ratio = round(len(source_sentence) / len(simplify_sentence), 3)
        compression_ratios.append(compression_ratio)

        deleted_characters = len(source_sentence) - len(simplify_sentence)
        deletions_proportion = round(deleted_characters / len(source_sentence), 3)
        deletions_proportions.append(deletions_proportion)


    results_df = pd.DataFrame({
        "snt_id": snt_ids,
        "fkgl_source": fkgl_source,
        "fkgl_simplify": fkgl_simplify,
        "Compression Ratio": compression_ratios,
        "Deletion Proportion": deletions_proportions
    }) 

    # Write DataFrame to TSV file
    results_df.to_csv("Task3_metrics_comparison.tsv", sep='\t', index=False)


def main():
    file_path = 'simpletext_task3_train_1.tsv'
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)

    # Simplify the query texts
    simplified_texts = simplify_text(source_sentences)

    create_tsv(snt_ids, simplified_texts)

    simplified_snts = get_simplified_tsv_data()

    create_tsv_metrics(snt_ids, source_sentences, simplified_snts)

if __name__ == '__main__':
    main()