from transformers import LlamaTokenizer, LlamaModel
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the tokenizer and model
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaModel.from_pretrained(model_name)

# Input text and tokenization
text = "Hello, my name is Julia."
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

# Get token embeddings
embedding_layer = model.get_input_embeddings()
token_embeddings = embedding_layer(input_ids)

# Convert embeddings to numpy for visualization
embeddings_np = token_embeddings[0].detach().numpy()  # Remove batch dimension and convert to numpy

# Dimensionality reduction with PCA to 3 components
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings_np)

# Convert IDs to tokens for labeling
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], marker='o', color='blue')

# Annotate each point with the corresponding token
for i, token in enumerate(tokens):
    ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], token, fontsize=12)

# Set labels
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.set_zlabel("PCA Component 3")
ax.set_title("Token Embeddings in 3D Space (PCA)")
plt.show()

# Save the plot as an image file
plt.savefig("token_embeddings_plot_3d.png")



