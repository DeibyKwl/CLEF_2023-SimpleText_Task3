import pandas as pd
import torch
import sys

import textstat
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu

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

    results_df.to_csv("Task3_TopGap_mistral_4.tsv", sep='\t', index=False)

def get_simplified_tsv_data():
    data = pd.read_csv('Task3_TopGap_mistral_4.tsv', sep='\t')
    return data['simplified_snt'].tolist()

def get_qrels_tsv_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['simplified_snt'].tolist()

def create_tsv_metrics(snt_ids, source_sentences, simplified_snts, qrels_snts):

    fkgl_source = []
    fkgl_simplify = []
    sari_scores = []
    bleu_scores = []
    compression_ratios = []

    for source_sentence, simplify_sentence, qrels_sentence in zip(source_sentences, simplified_snts, qrels_snts):
        fkgl_sentence1 = round(textstat.flesch_kincaid_grade(source_sentence), 3)
        fkgl_sentence2 = round(textstat.flesch_kincaid_grade(simplify_sentence), 3)
        fkgl_source.append(fkgl_sentence1)
        fkgl_simplify.append(fkgl_sentence2)

        sari = load("sari")
        sari_score = sari.compute(sources=[source_sentence], predictions=[simplify_sentence], references=[[qrels_sentence]])
        sari_scores.append(round(sari_score['sari'], 3))

        reference = [ref[0].split() for ref in [[qrels_sentence]]]
        candidate = simplify_sentence.split()
        bleu_score = round(sentence_bleu(reference, candidate), 3)
        bleu_scores.append(bleu_score)

        compression_ratio = round(len(simplify_sentence) / len(source_sentence), 3)
        compression_ratios.append(compression_ratio)

    results_df = pd.DataFrame({
        "snt_id": snt_ids,
        "fkgl_source": fkgl_source,
        "fkgl_simplify": fkgl_simplify,
        "sari": sari_scores,
        "bleu": bleu_scores,
        "Compression Ratio": compression_ratios
    }) 

    results_df.to_csv("Task3_metrics_comparison_mistral_4.tsv", sep='\t', index=False)


def main():
    if len(sys.argv) != 3:
        print('Usage: python Task3_Llama2_baseline <filename1> <filename2>')
        return
    file_path = sys.argv[1]
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)

    simplified_texts = simplify_text(source_sentences)

    create_tsv(snt_ids, simplified_texts)
    simplified_snts = get_simplified_tsv_data()

    file_path2 = sys.argv[2]
    qrels_snts = get_qrels_tsv_data(file_path2)
    create_tsv_metrics(snt_ids, source_sentences, simplified_snts, qrels_snts)

if __name__ == '__main__':
    main()