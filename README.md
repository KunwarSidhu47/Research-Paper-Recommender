# Research Paper Recommender

**Research Paper Recommender** is a semantic search-based web application that helps users discover relevant academic research papers using natural language queries. It uses sentence embeddings to understand the meaning of the query and retrieve the most semantically similar papers based on their abstracts.

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-HF%20Spaces-blue?logo=huggingface)](https://kunwarsidhu-research-paper-recommender.hf.space)

---

## Features

- Semantic search using SentenceTransformers and cosine similarity  
- Highlights query keywords in abstracts  
- Filter by author name or paper title  
- Paginated results (top 5 at a time)

---

## Tech Stack

- **Interface**: Gradio  
- **ML Model**: SentenceTransformer (`all-MiniLM-L6-v2`)  
- **Similarity**: Cosine similarity (NumPy)  
- **Data**: Custom dataset (`filtered_papers.csv`, `paper_embeddings.npy`)  
- **Hosting**: Hugging Face Spaces

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/research-paper-recommender.git
cd research-paper-recommender
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Make Sure You Have These Files

- `filtered_papers.csv` – Paper metadata (title, author, summary)  
- `paper_embeddings.npy` – Pre-computed NumPy embeddings

### 4. Launch the App

```bash
python app.py
```

You’ll see:

```bash
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxx.gradio.live
```

Use the public link to share your app with others.

---

## How It Works

- User submits a query  
- Query is converted to an embedding using `all-MiniLM-L6-v2`  
- Cosine similarity is computed with paper vectors  
- Top matches are shown with highlighted summaries  
- Optional filters: author and title match

---

## Project Structure

```bash
.
├── app.py                  # Main Gradio application
├── filtered_papers.csv     # Paper metadata
├── paper_embeddings.npy    # NumPy embedding vectors
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

## Future Enhancements

- Save user search history  
- Use FAISS for large-scale search  
- Add export or bookmark feature  
- Deploy with CI/CD pipelines

---

## License

This project is licensed under the **MIT License**.

---

Live Demo: [https://kunwarsidhu-research-paper-recommender.hf.space/](https://kunwarsidhu-research-paper-recommender.hf.space/)

If you found this helpful, consider starring the repo on GitHub.
