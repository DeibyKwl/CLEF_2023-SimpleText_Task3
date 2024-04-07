# SimpleText Task 3: Text Simplification
This code will attempt to generate simplified versions of scientific sentences in order to make scientific literature more accessible to the wider public.

# Before
Install the required libraries before running the main file, `pip install -r requirements.txt`

# How to run
Run `python Task3_llama2_baseline <filename1> <filename2>`, filename1 and filename2 are the datasets provided by CLEF lab. filename1 must contain the training data while filename2 must be a corpus with simplified sentences. Both files must have the same number of sentences/rows.
The filename1 must be in tsv format, and it should contain the following headers: `query_id`, `query_text`, `doc_id`, `snt_id`, and `source_snt`. filename2 must also be in tsv format and it should have the next headers: `snt_id`, `simplified_snt`. 

Example run `python Task3_llama2_baseline simpletext_task3_train.tsv simpletext_task3_qrels.tsv`