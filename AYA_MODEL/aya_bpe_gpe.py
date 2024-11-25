from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd

# Load the tokenizer with BPE
model_name = "CohereForAI/aya-expanse-8b"  # Ensure the tokenizer uses BPE
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the languages
languages = ["en", "de", "ar", "ru", "be"]  # English, German, Arabic, Russian, Belorussian

# Function to apply Grapheme Pair Encoding (GPE)
def apply_gpe(text):
    """
    Preprocesses the input text to split it into grapheme pairs.
    Example: "hello" -> "he ll o"
    """
    return " ".join([text[i:i+2] for i in range(0, len(text), 2)])

# Function to tokenize texts using BPE with Grapheme Pair Encoding
def tokenize_with_gpe_bpe(texts, tokenizer):
    """
    Tokenizes a list of texts using Grapheme Pair Encoding (GPE) followed by a BPE-based tokenizer.
    Returns a dictionary containing:
        - tokenized_texts: List of tokenized texts.
        - token_counts: List of token counts for each text.
    """
    # Apply GPE preprocessing to each text
    gpe_texts = [apply_gpe(text) for text in texts]
    
    # Tokenize all GPE-processed texts
    tokenized_texts = [tokenizer.tokenize(text) for text in gpe_texts]
    
    # Compute token counts for each GPE-processed text
    token_counts = [len(tokens) for tokens in tokenized_texts]
    
    return {
        "tokenized_texts": tokenized_texts,
        "token_counts": token_counts
    }

# Step 1: Load and tokenize datasets for each language using GPE + BPE
loaded_datasets = {}
tokenized_datasets = {}
token_count_datasets = {}

for lang in languages:
    print(f"Loading and tokenizing dataset with GPE for language: {lang}")
    try:
        # Load the dataset and limit it to 1000 sentences
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
        texts = dataset_lang["text"][:1000]  # Limit to the first 1000 sentences
        loaded_datasets[lang] = texts  # Store raw dataset

        # Tokenize using GPE + BPE
        tokenized_data = tokenize_with_gpe_bpe(texts, tokenizer)
        tokenized_datasets[lang] = tokenized_data["tokenized_texts"]
        token_count_datasets[lang] = tokenized_data["token_counts"]
    except Exception as e:
        print(f"Error loading or tokenizing dataset for {lang}: {e}")

# Step 2: Calculate Maximum Compression Ratio
def calculate_compression_ratio(texts, token_counts):
    """
    Calculates the maximum compression ratio for a set of texts.
    Compression Ratio = Number of Characters / Number of Tokens
    """
    compression_ratios = [
        len(text) / token_count if token_count > 0 else 0
        for text, token_count in zip(texts, token_counts)
    ]
    return max(compression_ratios) if compression_ratios else 0

compression_results = []
for lang in languages:
    print(f"Calculating Maximum Compression Ratio for: {lang}")
    try:
        texts = loaded_datasets[lang]
        token_counts = token_count_datasets[lang]
        max_compression_ratio = calculate_compression_ratio(texts, token_counts)
        compression_results.append({"Language": lang, "Max Compression Ratio": max_compression_ratio})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Save Maximum Compression Ratio results
df_compression = pd.DataFrame(compression_results)
df_compression.to_csv("aya_gpe_bpe_max_compression_ratio.csv", index=False)
print("Results saved to aya_gpe_bpe_max_compression_ratio.csv")
print(df_compression)

# Step 3: Calculate Minimum Tokenization Parity
def calculate_min_tokenization_parity(token_counts):
    """
    Calculates the minimum tokenization parity for a set of token counts.
    Parity = Minimum Token Count / Maximum Token Count
    """
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    return min_tokens / max_tokens if max_tokens > 0 else 0

tokenization_parity_results = []
for lang in languages:
    print(f"Calculating Minimum Tokenization Parity for: {lang}")
    try:
        token_counts = token_count_datasets[lang]
        min_tokenization_parity = calculate_min_tokenization_parity(token_counts)
        tokenization_parity_results.append({"Language": lang, "Minimum Tokenization Parity": min_tokenization_parity})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

# Save Minimum Tokenization Parity results
df_parity = pd.DataFrame(tokenization_parity_results)
df_parity.to_csv("aya_gpe_bpe_min_tokenization_parity.csv", index=False)
print("Results saved to aya_gpe_bpe_min_tokenization_parity.csv")
print(df_parity)
