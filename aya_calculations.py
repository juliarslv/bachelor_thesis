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

# Function to calculate Maximum Compression Ratio
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
    return max(results)  # Maximum compression ratio for the language

# Analyze for selected languages
compression_results = []
for lang in languages:
    try:
        print(f"Processing language: {lang}")
        # Load only 1% of the dataset and limit to 100 entries for efficiency
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train[:1%]")
        texts = dataset_lang["text"][:100]  # Limit to first 100 entries
        
        # Calculate Maximum Compression Ratio
        max_compression_ratio = calculate_compression_ratio(texts, tokenizer)
        compression_results.append({"Language": lang, "Max Compression Ratio": max_compression_ratio})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Create a DataFrame for results
df_compression = pd.DataFrame(compression_results)

# Save the results to a CSV file
df_compression.to_csv("language_compression_results_with_pretokenization_limited.csv", index=False)
print("Results saved to language_compression_results_with_pretokenization_limited.csv")
print(df_compression)


# Function to calculate Minimum Tokenization Parity
def calculate_min_tokenization_parity(texts, tokenizer):
    """
    Calculates the minimum tokenization parity for a set of texts.
    Parity = Minimum Token Count / Maximum Token Count
    """
    token_counts = [len(pretokenize(text, tokenizer)) for text in texts]
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    return min_tokens / max_tokens if max_tokens > 0 else 0


# Calculate Minimum Tokenization Parity for each language
tokenization_parity_results = []
for lang in languages:
    try:
        print(f"Calculating Minimum Tokenization Parity for: {lang}")
        # Use the same 1% dataset already loaded
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train[:1%]")
        texts = dataset_lang["text"][:100]  # Limit to first 100 entries for consistency
        
        # Calculate Minimum Tokenization Parity
        min_tokenization_parity = calculate_min_tokenization_parity(texts, tokenizer)
        tokenization_parity_results.append({"Language": lang, "Minimum Tokenization Parity": min_tokenization_parity})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Create a DataFrame for results
df_parity = pd.DataFrame(tokenization_parity_results)

# Save the results to a CSV file
df_parity.to_csv("minimum_tokenization_parity_1percent.csv", index=False)
print("Results saved to minimum_tokenization_parity_1percent.csv")
print(df_parity)
