import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import re

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import gradio as gr

load_dotenv()

# ── Load books ──────────────────────────────────────────────────────────────
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"].str.replace("http://", "https://") + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "No_Cover.jpg",
    books["large_thumbnail"],
)

# ── Load embeddings and vector store ────────────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Load pre-built DB — do NOT rebuild on every startup
db_books = Chroma(persist_directory="book_db", embedding_function=embeddings)


# ── Recommendation logic ─────────────────────────────────────────────────────
def retrieve_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(re.search(r'\d+', rec.page_content.split()[0]).group()) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_book(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        truncated_description = " ".join(row["description"].split()[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            author_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            author_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            author_str = row["authors"]

        caption = f"{row['title']} by {author_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# ── UI ───────────────────────────────────────────────────────────────────────
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("Book Recommendations")
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book",
            placeholder="e.g., A Story about forgiveness",
        )
        category_dropdown = gr.Dropdown(choices=categories, value="All", label="Select a category")
        tone_dropdown = gr.Dropdown(choices=tones, value="All", label="Select a tone")
        submit_button = gr.Button("Find Recommendations")
    gr.Markdown("Recommendations")
    output = gr.Gallery(label="Recommend Books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_book,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)
