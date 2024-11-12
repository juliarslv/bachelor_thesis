'''
1. pip install transformers torch

2. pip install sentencepiece

Accsess to hugginh face is needed

3. huggingface-cli login
 then --- create and enter token to bash

'''


tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "Hello, my name is Julia."

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)

print(token_embeddings)


print(token_embeddings.shape)


