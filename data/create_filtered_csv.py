import pandas as pd
import os

# Load the dataset
data_path = os.path.join("data", "arxiv_scientific_dataset.csv")
df = pd.read_csv(data_path)

# Filter papers from "Computer Science - Computation and Language" → cs.CL
filtered_df = df[df['category_code'].str.contains("cs.CL", na=False)]

# Save filtered result
filtered_df.to_csv("filtered_papers.csv", index=False)

print("✅ Filtered CSV created successfully.")

