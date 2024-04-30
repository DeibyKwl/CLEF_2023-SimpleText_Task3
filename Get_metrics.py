import pandas as pd
import sys

import textstat
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu


def get_source_tsv_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['source_snt'].tolist()
    
def get_qrels_tsv_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    return data['simplified_snt'].tolist()

def get_simplified_tsv_data(result_paths):
    results = []
    for path in result_paths:
        file_name = path[:-4] # Get file name, and remove the .tsv extension from name
        result = pd.read_csv(path, sep='\t')
        results.append([file_name, result['simplified_snt'].tolist()])
    return results

def create_metrics(source_snt, simplified_snts, qrels_snts):

    model_results = []
    fkgl_source_avg = []
    fkgl_simplify_avg = []
    sari_scores_avg = []
    bleu_scores_avg = []
    compression_ratios_avg = []

    for simplified_snt in simplified_snts: # Iterate over models simplified sentences

        fkgl_source = []
        fkgl_simplify = []
        sari_scores = []
        bleu_scores = []
        compression_ratios = []

        for source_sentence, simplify_sentence, qrels_sentence in zip(source_snt, simplified_snt[1], qrels_snts):
            print(source_sentence)
            fkgl_sentence1 = round(textstat.flesch_kincaid_grade(source_sentence), 2)
            fkgl_sentence2 = round(textstat.flesch_kincaid_grade(simplify_sentence), 2)
            fkgl_source.append(fkgl_sentence1)
            fkgl_simplify.append(fkgl_sentence2)

            sari = load("sari")
            sari_score = sari.compute(sources=[source_sentence], predictions=[simplify_sentence], references=[[qrels_sentence]])
            sari_scores.append(round(sari_score['sari'], 2))

            reference = [ref[0].split() for ref in [[qrels_sentence]]]
            candidate = simplify_sentence.split()
            bleu_score = round(sentence_bleu(reference, candidate), 2)
            bleu_scores.append(bleu_score)

            compression_ratio = round(len(simplify_sentence) / len(source_sentence), 2)
            compression_ratios.append(compression_ratio)

        model_results.append(simplified_snt[0])
        fkgl_source_avg.append(round(sum(fkgl_source) / len(fkgl_source), 2))
        fkgl_simplify_avg.append(round(sum(fkgl_simplify) / len(fkgl_simplify), 2))
        sari_scores_avg.append(round(sum(sari_scores) / len(sari_scores), 2))
        bleu_scores_avg.append(round(sum(bleu_scores) / len(bleu_scores), 2))
        compression_ratios_avg.append(round(sum(compression_ratios) / len(compression_ratios), 2))

    results_df = pd.DataFrame({
        "model": model_results,
        "fkgl_source": fkgl_source_avg,
        "fkgl_simplify": fkgl_simplify_avg,
        "sari": sari_scores_avg,
        "bleu": bleu_scores_avg,
        "Compression Ratio": compression_ratios_avg
    }) 

    results_df.to_csv("Metrics_Results.tsv", sep='\t', index=False)


def main():
    if len(sys.argv) < 4:
        print('Usage: python Get_metrics.py <filename1> <filename2>...')
        return
    
    source_path = sys.argv[1]

    # TODO: make sure qrel_path have the qrels name in it
    qrel_path = sys.argv[2]
    result_paths = [path for path in sys.argv[3:]]

    source_snt = get_source_tsv_data(source_path)
    qrels_snts = get_qrels_tsv_data(qrel_path)
    simplified_snts = get_simplified_tsv_data(result_paths)

    create_metrics(source_snt, simplified_snts, qrels_snts)


if __name__ == '__main__':
    main()