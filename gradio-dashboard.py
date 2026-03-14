import pandas as pd
import numpy as np
from dotenv import load_dotenv
from gradio.themes.builder_app import themes
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import vertexai
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr


load_dotenv()


books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "No_Cover.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=6000, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

vertexai.init(project="llm-vertex-489903", location="us-central1")
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

# Much smaller batch size due to token limit
BATCH_SIZE = 10

# Create DB with first batch
db_books = Chroma.from_documents(documents[:BATCH_SIZE], embedding=embeddings)

# Add remaining in small batches
for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    db_books.add_documents(batch)
    time.sleep(1)  # avoid rate limiting
    print(f"Processed {min(i + BATCH_SIZE, len(documents))}/{len(documents)} docs")


def retrieve_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)
    books_list = [int(rec.page_content.split()[0].strip()) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    if tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    if tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    if tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    if tone == "Sad":
        book_recs.sort_values(by="sad", ascending=False, inplace=True)

    return book_recs

def recommend_book(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + ...

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            author_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            author_str = f"{','.join(authors_split[:-1])}, and {authors_split[:1]}"
        else:
            author_str = row["authors"]

        caption = f"{row['title']} by {author_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad "]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("Book Recommendations")
    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book",
            placeholder="e.g., A Story about forgiveness",
        )
        category_dropdown = gr.Dropdown(choices=categories, value="All", label="Select a category")
        tone_dropdown = gr.Dropdown(choices=tones, value="All", label="Select a tone")
        submit_botton = gr.Button("Find Recommendations")
    gr.Markdown("Recommendations")
    output = gr.Gallery(label="Recommend Books", columns=8, rows=2)

    submit_botton.click(fn=recommend_book,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()