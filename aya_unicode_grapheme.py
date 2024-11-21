from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import pandas as pd

# Load the model and tokenizer
model_name = "CohereForAI/aya-expanse-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Updated phrases
phrases = {
    "English": "Thanks",       # English
    "German": "Danke",         # German
    "Arabic": "شكراً",         # Arabic
    "Russian": "Спасибо",      # Russian
    "Belorussian": "Дзякуй"    # Belorussian
}

# Function to analyze text: UTF-8 bytes, Unicode codepoints, and grapheme clusters
def analyze_text_utf8(text):
    utf8_bytes = text.encode('utf-8')  # Encode text as UTF-8
    # Combine letters with their Unicode codepoints
    codepoints = [f"{char} (U+{ord(char):04X})" for char in text]
    graphemes = [char for char in text]  # Grapheme clusters as visible letters
    return utf8_bytes, codepoints, graphemes

# Function for pretokenization using the model's tokenizer
def pretokenize_with_graphemes(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    graphemes = [unicodedata.normalize('NFC', token) for token in tokens]
    return graphemes

# Perform analysis and comparison
results = {}
for lang, phrase in phrases.items():
    # Analyze the phrase
    utf8_bytes, codepoints, graphemes = analyze_text_utf8(phrase)
    # Pretokenize using the model's tokenizer
    pretokenized_graphemes = pretokenize_with_graphemes(phrase, tokenizer)
    results[lang] = {
        "Phrase": phrase,
        "UTF-8 Bytes": utf8_bytes,
        "Unicode Codepoints": codepoints,
        "Grapheme Clusters": graphemes,  # Visible letters
        "Pretokenized Graphemes": pretokenized_graphemes
    }

# Prepare data for output
table_data = {"": ["Unicode Codepoint (with letters)", "Grapheme Clusters"]}
for lang, data in results.items():
    table_data[lang] = [", ".join(data["Unicode Codepoints"]), ", ".join(data["Grapheme Clusters"])]

# Create DataFrame for the table
comparison_df = pd.DataFrame(table_data)

# Save the table to a CSV file
comparison_df.to_csv("table_unicode_graphemes.csv", index=False)

# Print the table for review
print(comparison_df)
