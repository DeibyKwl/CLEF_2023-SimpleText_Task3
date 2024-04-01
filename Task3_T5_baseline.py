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
    
    for index, row in data.iterrows():
        query_text = row['query_text']
        snt_id = row['snt_id']
        source_snt = row['source_snt']

        query_texts.append(query_text)
        snt_ids.append(snt_id)
        source_sentences.append(source_snt)

    return snt_ids, source_sentences, query_texts

# def simplify_text(sentences):
#     simplified_texts = []

#     for sentence in sentences:
#         print(sentence)
#         # Preprocess the input sentence
#         input_text = "Simplify: " + sentence
#         input_ids = tokenizer.encode(input_text, return_tensors="pt")

#         # Generate simplified text
#         output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
#         simplified_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

#         simplified_texts.append(simplified_text)

#     return simplified_texts

def summarize_text(sentences):

    summaries = []

    for sentence in sentences:
        inputs = tokenizer.encode(
            "summarize: " + sentence,
            return_tensors='pt',
            max_length=50,
            truncation=True
        )
    
        # Generate the summary
        summary_ids = model.generate(
            inputs,
            max_length=512,
            num_beams=5,
            # early_stopping=True,
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
 
    # Decode and return the summary
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
    simplified_texts = summarize_text(source_sentences)
    create_tsv(snt_ids, source_sentences)



if __name__ == '__main__':
    main()