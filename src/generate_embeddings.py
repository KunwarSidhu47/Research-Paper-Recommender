import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load filtered papers
data_path = os.path.join("data", "filtered_papers.csv")
df = pd.read_csv(data_path)

# Combine title + summary for better context
texts = (df['title'] + ". " + df['summary']).tolist()

# Load a pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Save embeddings as a .npy file
np.save(os.path.join("data", "paper_embeddings.npy"), embeddings)

print("✅ Embeddings generated and saved.")
