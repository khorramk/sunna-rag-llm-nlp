import os
import torch
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

app = FastAPI(title="Hadith Semantic Search Engine (meeAtif Dataset)")

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_FILE = "hadith_embeddings_meeatif.pt"
model = None
hadith_records = []
hadith_embeddings = None

@app.on_event("startup")
def startup_event():
    global model, hadith_records, hadith_embeddings
    print("🚀 Initializing application elements...")
    
    model = SentenceTransformer(MODEL_NAME)
    
    print("📥 Downloading meeAtif/hadith_datasets (Sahih al-Bukhari)...")
    # Correct way to pass file targets down to the Hugging Face loading engine
    dataset = load_dataset("meeAtif/hadith_datasets", data_files={"train": "Sahih al-Bukhari.json"}, split="train")
    
    # Process the custom JSON layout keys inside the meeAtif repo
    print("⚙️ Processing database metadata mapping...")
    hadith_records = []
    for item in dataset:
        if item.get('English_Text'):
            hadith_records.append({
                "text": item['English_Text'].strip(),
                "book": item.get('Book', 'Sahih al-Bukhari'),
                "chapter": item.get('Chapter_Title_English', 'General Chapters'),
                "number": item.get('In-book reference', 'N/A'), # Using their unified reference key string
                "grade": item.get('Grade', 'Sahih') # Includes authentification grade if available
            })
            
    print(f"✅ Loaded {len(hadith_records)} Hadiths successfully.")
    texts_to_encode = [record['text'] for record in hadith_records]

    if not os.path.exists(EMBEDDINGS_FILE):
        print("🧠 Computing vectors for the dataset (takes a brief moment on first load)...")
        hadith_embeddings = model.encode(texts_to_encode, show_progress_bar=True, convert_to_tensor=True)
        torch.save(hadith_embeddings, EMBEDDINGS_FILE)
        print("💾 Array vectors successfully cached onto disk storage!")
    else:
        print("📂 Existing vectorized tensor cache loaded from file.")
        hadith_embeddings = torch.load(EMBEDDINGS_FILE, weights_only=True)
    print("⚡ Server is completely ready to accept browser queries!")

@app.get("/api/search")
def search_hadiths(q: str = Query(..., description="The search query text"), limit: int = 5):
    # Ensure the vectors are fully loaded before handling calls
    if hash(hadith_embeddings) is None or not q:
        return {"results": []}
        
    query_embedding = model.encode(q, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, hadith_embeddings)[0]
    
    top_results = torch.topk(cos_scores, k=min(limit, len(hadith_records)))
    
    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        matched_item = hadith_records[idx.item()]
        results.append({
            "text": matched_item['text'],
            "score": round(score.item(), 4),
            "book": matched_item['book'],
            "number": matched_item['number'],
            "chapter": matched_item['chapter'],
            "grade": matched_item['grade']
        })
        
    return {"results": results}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Hadith Semantic Search</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-100 text-slate-800 font-sans min-h-screen">
        <div class="max-w-4xl mx-auto px-4 py-12">
            <header class="text-center mb-12">
                <h1 class="text-4xl font-extrabold text-slate-900 mb-2">Hadith Semantic Search</h1>
                <p class="text-slate-600">Query text collections using local NLP sentence-transformers.</p>
            </header>

            <div class="bg-white p-6 rounded-xl shadow-md mb-8">
                <div class="flex gap-4">
                    <input type="text" id="query" placeholder="e.g., deeds and intentions, purity before prayer..." 
                           class="flex-1 px-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500 text-lg"
                           onkeyup="if(event.key === 'Enter') performSearch()">
                    <button onclick="performSearch()" class="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-6 py-3 rounded-lg transition-colors">
                        Search
                    </button>
                </div>
            </div>

            <div id="loading" class="hidden text-center py-6 text-slate-500 font-medium animate-pulse">Searching vector models...</div>
            <div id="results" class="space-y-6"></div>
        </div>

        <script>
            async function performSearch() {
                const query = document.getElementById('query').value.trim();
                const resultsContainer = document.getElementById('results');
                const loading = document.getElementById('loading');
                
                if (!query) return;

                loading.classList.remove('hidden');
                resultsContainer.innerHTML = '';

                try {
                    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
                    const data = await response.json();
                    
                    loading.classList.add('hidden');
                    
                    if (!data.results || data.results.length === 0) {
                        resultsContainer.innerHTML = '<div class="text-center text-slate-500 py-6">No matching citations found.</div>';
                        return;
                    }

                    data.results.forEach(item => {
                        const scorePercentage = Math.round(item.score * 100);
                        const card = document.createElement('div');
                        card.className = "bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden";
                        
                        card.innerHTML = `
                            <div class="bg-slate-50 border-b border-slate-200 px-6 py-3.5 flex flex-wrap justify-between items-center gap-2">
                                <div class="font-semibold text-slate-700 text-sm">
                                    📖 <span class="text-emerald-700 font-bold">${item.book}</span> 
                                    <span class="text-slate-400 mx-1.5">|</span> ${item.number}
                                </div>
                                <div class="flex items-center gap-2">
                                    ${item.grade ? `<span class="bg-blue-50 text-blue-700 text-xs font-semibold px-2 py-0.5 rounded border border-blue-200">${item.grade}</span>` : ''}
                                    <span class="bg-emerald-100 text-emerald-800 text-xs font-bold px-2.5 py-1 rounded-full">${scorePercentage}% Match</span>
                                </div>
                            </div>
                            <div class="p-6">
                                <div class="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">${item.chapter}</div>
                                <p class="text-slate-800 leading-relaxed font-serif text-base">${item.text}</p>
                            </div>
                        `;
                        resultsContainer.appendChild(card);
                    });
                } catch (error) {
                    loading.classList.add('hidden');
                    resultsContainer.innerHTML = '<div class="text-center text-red-500 py-6">An issue occurred handling the interface response stream.</div>';
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    """
