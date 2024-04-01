from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# Load pre-trained T5 model and tokenizer
model_name = "google-t5/t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)


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

    summaries = []
    
    for sentence in sentences:
        # Preprocess the input by incorporating query words
        input_text = "summarize: " + sentence
        inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
        
        # Generate the summary
        summary_ids = model.generate(
            inputs,
            max_length=150,
            num_beams=5,
            # early_stopping=True,
        )
        
        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
    
    return summaries

def create_tsv(snt_ids, summaries):
    results_df = pd.DataFrame({
        "run_id": ["TopGap_task3_run1"] * len(snt_ids),
        "manual": [0] * len(snt_ids),
        "snt_id": snt_ids,
        "simplified_snt": summaries
    })

    # Write DataFrame to TSV file
    results_df.to_csv("Task3_TopGap.tsv", sep='\t', index=False)


def main():
    file_path = 'simpletext_task3_train.tsv'
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)

    # Simplify the query texts
    simplified_texts = simplify_text(source_sentences)
    create_tsv(snt_ids, simplified_texts)



if __name__ == '__main__':
    main()