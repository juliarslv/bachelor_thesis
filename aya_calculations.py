from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

# Load the tokenizer
model_name = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the languages
languages = ["en", "de", "ar", "ru", "be"]  # English, German, Arabic, Russian, Belorussian

# Pre-tokenization function
def pretokenize(text, tokenizer):
    """
    Applies pre-tokenization to the input text using the tokenizer's tokenize function.
    """
    return tokenizer.tokenize(text)

# Step 1: Load dataset for each language
loaded_datasets = {}
for lang in languages:
    print(f"Loading dataset for language: {lang}")
    try:
        # Load the dataset and limit it to 1000 sentences
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
        texts = dataset_lang["text"][:1000]  # Limit to the first 1000 sentences
        loaded_datasets[lang] = texts  # Store dataset for later calculations
    except Exception as e:
        print(f"Error loading dataset for {lang}: {e}")

# Step 2: Calculate Maximum Compression Ratio
def calculate_compression_ratio(texts, tokenizer):
    """
    Calculates the maximum compression ratio for a set of texts.
    Compression Ratio = Number of Characters / Number of Tokens
    """
    results = []
    for text in texts:
        pre_tokens = pretokenize(text, tokenizer)  # Use pre-tokenization
        num_characters = len(text)
        num_tokens = len(pre_tokens)
        compression_ratio = num_characters / num_tokens if num_tokens > 0 else 0
        results.append(compression_ratio)
    return max(results)  # Return the maximum compression ratio


compression_results = []
for lang, texts in loaded_datasets.items():
    print(f"Calculating Maximum Compression Ratio for: {lang}")
    try:
        max_compression_ratio = calculate_compression_ratio(texts, tokenizer)
        compression_results.append({"Language": lang, "Max Compression Ratio": max_compression_ratio})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Save Maximum Compression Ratio results
df_compression = pd.DataFrame(compression_results)
df_compression.to_csv("max_compression_ratio.csv", index=False)
print("Results saved to max_compression_ratio.csv")
print(df_compression)

# Step 3: Calculate Minimum Tokenization Parity
def calculate_min_tokenization_parity(texts, tokenizer):
    """
    Calculates the minimum tokenization parity for a set of texts.
    Parity = Minimum Token Count / Maximum Token Count
    """
    token_counts = [len(pretokenize(text, tokenizer)) for text in texts]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    return min_tokens / max_tokens if max_tokens > 0 else 0

tokenization_parity_results = []
for lang, texts in loaded_datasets.items():
    print(f"Calculating Minimum Tokenization Parity for: {lang}")
    try:
        min_tokenization_parity = calculate_min_tokenization_parity(texts, tokenizer)
        tokenization_parity_results.append({"Language": lang, "Minimum Tokenization Parity": min_tokenization_parity})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Save Minimum Tokenization Parity results
df_parity = pd.DataFrame(tokenization_parity_results)
df_parity.to_csv("min_tokenization_parity.csv", index=False)
print("Results saved to min_tokenization_parity.csv")
print(df_parity)
