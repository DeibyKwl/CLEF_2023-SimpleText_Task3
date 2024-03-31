import pandas as pd
import torch


###################################################################################
# Load model directly
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
    
    print(data)

    return snt_ids, source_sentences, query_texts



def main():
    file_path = 'simpletext_task3_train.tsv'
    snt_ids, source_sentences, query_texts = get_tsv_data(file_path)
    prompt = 'Simplify the following text'

if __name__ == '__main__':
    main()