import gradio as gr
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv('filtered_papers.csv')
embeddings = np.load('paper_embeddings.npy')

def search_papers(query, author_filter="", title_filter=""):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    df["score"] = scores

    filtered_df = df.copy()
    if author_filter:
        filtered_df = filtered_df[filtered_df["authors"].str.contains(author_filter, case=False, na=False)]
    if title_filter:
        filtered_df = filtered_df[filtered_df["title"].str.contains(title_filter, case=False, na=False)]

    top_df = filtered_df.sort_values(by="score", ascending=False).head(10)

    results = []
    for _, paper in top_df.iterrows():
        summary = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", paper["summary"], flags=re.IGNORECASE)
        result_block = f"""
        <div style='padding: 1em; margin-bottom: 1em; border: 1px solid #444; border-radius: 8px; background-color: var(--secondary-background);'>
            <h3 style='margin-bottom: 0.5em;'>{paper['title']}</h3>
            <p style='font-size: 0.9em; color: gray;'>Author: {paper['authors']}</p>
            <p>{summary}</p>
        </div>
        """
        results.append(result_block)

    return "\n".join(results)

with gr.Blocks(theme=gr.themes.Soft(), css="body { background-color: var(--background-color); color: var(--text-color); }") as demo:
    gr.Markdown("## 🔍 Research Paper Recommender (Semantic Search Engine)")
    with gr.Row():
        query = gr.Textbox(label="Enter your query", placeholder="e.g. reinforcement learning in robotics", lines=1)
        author = gr.Textbox(label="Filter by Author (optional)", placeholder="e.g. John Doe", lines=1)
        title = gr.Textbox(label="Filter by Title (optional)", placeholder="e.g. Deep Learning Methods", lines=1)
    with gr.Row():
        search_btn = gr.Button("Search")
        clear_btn = gr.Button("Clear")
    output = gr.HTML()

    def clear_fields():
        return "", "", "", ""

    search_btn.click(fn=search_papers, inputs=[query, author, title], outputs=output)
    clear_btn.click(fn=clear_fields, inputs=[], outputs=[query, author, title, output])

demo.launch(server_name="0.0.0.0")
