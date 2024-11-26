from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import re

# Load the tokenizer for the mGPT model
model_name = "ai-forever/mGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the languages
languages = ["en", "de", "ar", "ru", "be"]  # English, German, Arabic, Russian, Belarusian

# Function to apply Grapheme Pair Encoding (GPE)
def apply_gpe(text):
    """
    Preprocesses the input text to split it into grapheme pairs.
    Example: "hello" -> "he ll o"
    """
    return " ".join([text[i:i+2] for i in range(0, len(text), 2)])

# Function to tokenize using GPE + mGPT Tokenizer
def tokenize_with_gpe(texts, tokenizer):
    """
    Tokenizes a list of texts using Grapheme Pair Encoding (GPE) followed by mGPT-based tokenization.
    Returns a dictionary containing:
        - tokenized_texts: List of tokenized texts.
        - token_counts: List of token counts for each text.
    """
    gpe_texts = [apply_gpe(text) for text in texts]
    tokenized_texts = [tokenizer.tokenize(text) for text in gpe_texts]
    token_counts = [len(tokens) for tokens in tokenized_texts]
    return {
        "tokenized_texts": tokenized_texts,
        "token_counts": token_counts
    }

# Regex-based sentence tokenizer
def split_into_sentences(text):
    """
    Splits a text into sentences using a regex-based approach.
    """
    sentence_endings = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(\s|\n)'
    sentences = re.split(sentence_endings, text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Step 1: Load and tokenize datasets for each language
loaded_sentences = {}
tokenized_datasets = {}
token_count_datasets = {}

for lang in languages:
    print(f"Loading and tokenizing dataset for language: {lang}")
    try:
        # Load the dataset and extract the text
        dataset_lang = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train")
        documents = dataset_lang["text"][:1000]  # Limit to the first 1000 documents
        
        # Split documents into sentences using regex
        sentences = []
        for doc in documents:
            sentences.extend(split_into_sentences(doc))
        
        # Limit to 1000 sentences
        sentences = sentences[:1000]
        loaded_sentences[lang] = sentences  # Store raw sentences

        # Tokenize using GPE + mGPT Tokenizer
        tokenized_data = tokenize_with_gpe(sentences, tokenizer)
        tokenized_datasets[lang] = tokenized_data["tokenized_texts"]
        token_count_datasets[lang] = tokenized_data["token_counts"]
    except Exception as e:
        print(f"Error loading or tokenizing dataset for {lang}: {e}")

# Display one example sentence and its tokenized form for each language
for lang in languages:
    print(f"\nExample for language: {lang}")
    try:
        if lang in tokenized_datasets:
            example_sentence = loaded_sentences[lang][0]
            gpe_sentence = apply_gpe(example_sentence)
            tokenized_example = tokenized_datasets[lang][0]
            print(f"Original Sentence: {example_sentence}")
            print(f"GPE Sentence: {gpe_sentence}")
            print(f"Tokenized Sentence: {tokenized_example}")
        else:
            print(f"No tokenized data available for {lang}")
    except Exception as e:
        print(f"Error displaying example for {lang}: {e}")

# Step 2: Calculate Maximum Compression Ratio
def calculate_max_compression_ratio(texts, token_counts):
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
        texts = loaded_sentences[lang]
        token_counts = token_count_datasets[lang]
        max_compression_ratio = calculate_max_compression_ratio(texts, token_counts)
        compression_results.append({"Language": lang, "Max Compression Ratio": max_compression_ratio})
    except Exception as e:
        print(f"Error processing {lang}: {e}")

df_compression = pd.DataFrame(compression_results)
df_compression.to_csv("mgpt_gpe_max_compression_ratio.csv", index=False)
print("Results saved to mgpt_gpe_max_compression_ratio.csv")
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

df_parity = pd.DataFrame(tokenization_parity_results)
df_parity.to_csv("mgpt_gpe_min_tokenization_parity.csv", index=False)
print("Results saved to mgpt_gpe_min_tokenization_parity.csv")
print(df_parity)
