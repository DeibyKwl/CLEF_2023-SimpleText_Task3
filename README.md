﻿# SimpleText Task 3: Text Simplification
This code will attempt to generate simplified versions of scientific sentences in order to make scientific literature more accessible to the wider public.

# Before
Install the required libraries before running the main file, `pip install -r requirements.txt`

# Setting Huggingface access
To run any of the models, you will have to go to huggingface and gain access to the following models: 
Llama 3, Llama 2, and/or Mistral. After having been granted access to these models, you must run `huggingface-cli login` in you console, and then use the token from your huggingface account to log in.

# How to run Models
First choose the model that you want to run, your options mistral, llama 2, and llama 3. Let say you want to run Llama 3, then you run in the console `python Task3_Llama3.py <filename>`, filename is the datasets provided by CLEF lab. The filename must be in tsv format, and it should contain the following headers: `query_id`, `query_text`, `doc_id`, `snt_id`, and `source_snt`. 

Example run `python Task3_Llama3.py simpletext_task3_train.tsv`

Running these models will created a tsv file with the simplified sentences.

# How to run Metrics
To get the metrics from a models result, you must run `python Get_metrics.py <filename1> <filename2> <filename3> ...` where filename1 is the training dataset provided by CLEF lab as previously mentioned. filename2 is the qrels file, also provided by CLEF lab. filename2 must be in tsv format and it should have the next headers: snt_id, simplified_snt. filename3 is the model result in tsv format from the previous runs; `python Task3_Llama3.py simpletext_task3_train.tsv` will get you the file `Task3_TopGap_Llama3.tsv` which is a file we can use for filename3. These script, `Get_metrics.py`, allows for multiple models result, so we could add a filename4 if we wanted to also include another model simplifications result, and if we have more, we could have a filename5. This will end up creating a tsv file called `Metrics_Results.tsv` with all the metrics results.

Example run for 3 models result `python Get_metrics.py simpletext_task3_train.tsv simpletext_task3_qrels.tsv Task3_TopGap_Llama2.tsv Task3_TopGap_Llama3.tsv Task3_TopGap_mistral.tsv`

Example run for 2 models result `python Get_metrics.py simpletext_task3_train.tsv simpletext_task3_qrels.tsv Task3_TopGap_mistral.tsv Task3_TopGap_Llama3.tsv`

An example result of running this script with 3 models results will get us the following:
![Table_Metrics_Results](https://github.com/DeibyKwl/CLEF_2023-SimpleText_Task3/blob/main/Table_Metrics_Results.png)
