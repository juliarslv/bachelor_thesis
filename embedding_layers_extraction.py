from transformers import LlamaTokenizer, LlamaModel

max_memory = {
    "cpu": "4GB",  
    "mps": "4GB" 
}

model = LlamaModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",          
    offload_folder="/tmp",    
    offload_state_dict=True,   
    max_memory=max_memory      
)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "Hello, this is a test."

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)

print(token_embeddings)


print(token_embeddings.shape)