# 📚 Semantic Book Recommender

An intelligent, interactive book recommendation engine that goes beyond simple keyword matching. This application uses semantic search, sentiment analysis, and text classification to suggest books based on natural language queries, desired genres, and emotional tones.

**Live Demo:** [Try the App Here](https://eswar1108-semantic-book-recommender.hf.space)

---

## ✨ Features
* **Semantic Search:** Utilizes the Google Gemini API (`models/gemini-embedding-001`) to understand the deep context of user queries and match them against book descriptions.
* **Emotion & Tone Filtering:** Recommends books that fit a specific mood (Happy, Surprising, Angry, Suspenseful, Sad) based on custom sentiment analysis.
* **Smart Categorization:** Filters recommendations by specific genres and categories.
* **Interactive UI:** A clean, glass-themed web interface built with Gradio.
* **Vector Database:** High-performance similarity search powered by ChromaDB.

## 🛠 Tech Stack
* **Language:** Python 3.11
* **Machine Learning & NLP:** LangChain, Google Generative AI (Gemini)
* **Vector Database:** ChromaDB
* **Web Framework:** Gradio
* **Data Processing:** Pandas, NumPy
* **Deployment:** Docker / Hugging Face Spaces

## 📂 Project Structure
* `gradio-dashboard.py`: The main Gradio application script that loads the vector database and serves the UI.
* `BR_Notebook.ipynb`: Data cleaning, preprocessing, and exploratory data analysis.
* `sentiment-analysis.ipynb`: NLP pipeline to extract emotional tones from book descriptions.
* `text-classification.ipynb`: Categorization pipeline for filtering books by genre.
* `Vector_search.ipynb`: Notebook demonstrating the creation of the vector database using Gemini embeddings.
* `Dockerfile`: Container configuration for easy deployment.
* `requirements.txt`: Python project dependencies.

---

## 🚀 How to Run Locally

You can run this project locally using a virtual environment or via Docker. 

### Prerequisites
1. You must have a Google API Key to use Gemini embeddings. 
2. Create a `.env` file in the root directory and add your key:
   ```env
   GOOGLE_API_KEY="your_api_key_here"
