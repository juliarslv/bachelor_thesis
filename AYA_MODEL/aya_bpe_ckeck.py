#Many pre-trained models use BPE as the tokenization method. An option to check it:


from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("CohereForAI/aya-expanse-8b")

# Check if 'merges_file' is in the tokenizer's initialization arguments
if "merges_file" in tokenizer.init_kwargs:
    print("The tokenizer uses BPE.")
    print("Merges file:", tokenizer.init_kwargs["merges_file"])
else:
    print("The tokenizer does not use BPE.")
