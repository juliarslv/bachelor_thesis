'''
1. pip install transformers torch

2. pip install sentencepiece

Accsess to hugginh face is needed

3. huggingface-cli login
 then --- create and enter token to bash

'''

from transformers import LlamaTokenizer, LlamaModel

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# Define the input text
text = "Hello, my name is Julia."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Get the embeddings
embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)

print(token_embeddings)
print(token_embeddings.shape)



