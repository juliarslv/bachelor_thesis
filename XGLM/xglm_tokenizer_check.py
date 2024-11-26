from transformers import AutoTokenizer

# Load the tokenizer for the XGLM model
model_name = "facebook/xglm-564M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Display the tokenizer type
print(type(tokenizer).__name__)



# >>> XGLMTokenizerFast