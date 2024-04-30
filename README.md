# SimpleText Task 3: Text Simplification
This code will attempt to generate simplified versions of scientific sentences in order to make scientific literature more accessible to the wider public.

# Before
Install the required libraries before running the main file, `pip install -r requirements.txt`

# Setting Huggingface access
To run any of the models, you will have to go to huggingface and gain access to the following models: 
Llama 3, Llama 2, and/or Mistral. After having been granted access to these models, you must run `huggingface-cli login` in you console, and then use the token from your huggingface account to log in.

# How to run
First choose the model that you want to run, your options mistral, llama 2, and llama 3. Let say you want to run Llama 3, then you run in the console `python Task3_Llama3.py <filename1>`, filename is the datasets provided by CLEF lab. The filename must be in tsv format, and it should contain the following headers: `query_id`, `query_text`, `doc_id`, `snt_id`, and `source_snt`. 

Example run `python Task3_Llama3.py simpletext_task3_train.tsv`


