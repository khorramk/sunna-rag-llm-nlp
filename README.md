# sunna-rag-llm-nlp
A NLP based search query against hadith Datasets


---

# Hadith Semantic Search Engine

A local, high-performance machine learning search engine built using **FastAPI** and **SentenceTransformers**. This application downloads prophetic traditions (Hadiths) from the `meeAtif/hadith_datasets` collection on Hugging Face and uses natural language processing (NLP) to search the text by **meaning, context, and intent**, rather than just rigid keyword matching.

---

## ✨ Features

* **Semantic Search:** Understands context. Querying *"donating to poor people"* will correctly return Hadiths talking about *"giving charity without delay"* even if the word "donating" or "poor" isn't explicitly used.
* **Smart Caching:** On the first launch, the app converts the entire dataset into mathematical vectors (embeddings) and caches them locally to your drive (`hadith_embeddings_meeatif.pt`). Subsequent app startups take less than a second.
* **Rich Metadata & Citations:** Displays complete verification cards containing the Book Title, Chapter name, Authenticity Grade (e.g., *Sahih*), and unified In-book reference numbering.
* **Lean Tech Stack:** Optimized entirely to run fast and efficiently on a standard consumer **CPU**. No expensive GPU cluster or external vector database cloud subscriptions required.

---

## 🛠️ Tech Stack

* **Backend:** Python, FastAPI, Uvicorn (ASGI Web Server)
* **ML Inference Engine:** SentenceTransformers (`all-MiniLM-L6-v2` model)
* **Dataset Source:** Hugging Face `datasets` (`meeAtif/hadith_datasets`)
* **Vector Math & Tensor Storage:** PyTorch
* **Frontend:** Single-page HTML5 UI beautifully styled with Tailwind CSS via CDN

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.9 or higher installed on your computer.

### 2. Installation

Clone or download this repository, navigate to your project folder using your terminal, and install the required dependencies:

```bash
pip install fastapi uvicorn sentence-transformers datasets torch

```

### 3. Running the Application

Because of environment path configurations on certain operating systems (like Windows), it is best to execute the server module directly via Python:

```bash
python -m uvicorn app:app --reload

```

### 4. Accessing the UI

1. Look at your terminal screen. The application will fetch the required dataset files and initialize the encoder model matrix.
2. Wait until you explicitly see the confirmation log output:
`⚡ Server is completely ready to accept browser queries!`
3. Open your browser and navigate to: **`http://127.0.0.1:8000`**
4. To explore or test the interactive OpenAPI backend documentation, visit: `http://127.0.0.1:8000/docs`

---

## 📂 Project Architecture

```markdown
├── app.py                     # Main application logic (FastAPI endpoints + HTML UI)
├── hadith_embeddings_meeatif.pt # Local binary cache of generated ML vectors (Created on 1st run)
└── README.md                  # Project documentation

```

---

## 💡 Changing the Data Collection Target

By default, the application indexes **Sahih al-Bukhari**. If you want to index alternative collections or combine multiple files, alter the `data_files` target block in the `startup_event()` function inside `app.py`:

```python
# To change to a single alternative collection (e.g., Sahih Muslim):
dataset = load_dataset("meeAtif/hadith_datasets", data_files={"train": "Sahih Muslim.json"}, split="train")

# Remember to delete your local 'hadith_embeddings_meeatif.pt' file if you switch collections so the app can rebuild its cache matrix!

```

---

## 📝 License

This project is open-source. All textual data properties belong to their respective uploaders on Sunnah.com and Hugging Face.
