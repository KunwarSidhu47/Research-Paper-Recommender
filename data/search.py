import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load the dataset and embeddings
df = pd.read_csv("data/filtered_papers.csv")
embeddings = np.load("data/paper_embeddings.npy")

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input query
query = input("Enter your research topic or query: ")

# Encode query
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute cosine similarities
cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

# Get top 5 most similar papers
top_k_indices = cosine_scores.argsort(descending=True)[:5]

# Print the results
print("\nTop relevant papers:\n")
for idx in top_k_indices:
    idx = int(idx)  # Ensure index is integer
    print(f"Title: {df.iloc[idx]['title']}")
    print(f"Summary: {df.iloc[idx]['summary']}\n")
