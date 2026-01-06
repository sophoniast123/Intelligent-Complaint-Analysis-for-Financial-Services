**Project: CFPB Complaint RAG Demo**

Brief demo that builds embeddings from consumer complaint narratives, stores them in a Chroma vector store, and provides a small RAG-based question-answering/chat interface.

**Repository Structure**
- **`data/`**: raw and processed datasets (CSV / parquet embeddings).
- **`notebook/`**: utility scripts and demo runners:
  - `app.py` — simple interactive chatbot using retrieval + OpenAI.
  - `embed_sample.py` — chunking + embedding generation pipeline (saves `sample_embeddings.parquet`).
  - `ingest.py` — ingest embeddings parquet into Chroma collection `complaints`.
  - `chunking.py` — helper to split long narratives into chunks.
  - `rag.py` — retrieval and answer-generation utilities (uses `sentence-transformers` + OpenAI).
- `requirements.txt` — pinned Python dependencies used by the project.

**Quick Setup**
1. Create and activate a virtual environment (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set required environment variables (example):

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

Note: `rag.py` currently expects `openai.api_key` to be set. Replace the placeholder or set the env var.

**Data**
- Place the original complaint CSV (or input dataset) inside `data/` — the repo includes `data/complaints.csv` as an entry point. The embedding scripts read/write parquet files such as `data/sample_embeddings.parquet` and `data/complaint_embeddings.parquet`.

**Typical Workflow**
1. Prepare a sample or full dataset and clean narratives (not included in this README).
2. Run `embed_sample.py` to chunk narratives and generate embeddings:

```bash
python notebook/embed_sample.py
```

This writes `data/sample_embeddings.parquet` containing `chunk_text` and `embedding` columns.

3. Build the vector DB with `ingest.py`:

```bash
python notebook/ingest.py
```

This creates/uses a Chroma collection named `complaints` and adds documents + metadata.

4. Start the simple CLI chatbot `app.py`:

```bash
python notebook/app.py
```

Type questions at the prompt; optionally provide a `product` filter when asked.

**Important Implementation Notes**
- Embeddings: `embed_sample.py` and `rag.py` use the `all-MiniLM-L6-v2` SentenceTransformer model.
- Vector DB: `ingest.py` uses `chromadb.Client()` and creates a collection named `complaints`.
- LLM calls: `rag.py` calls OpenAI ChatCompletion (model `gpt-4o-mini` in the code). Replace or configure as needed and keep your API key private.
- Chunking: `chunking.py` performs a simple sliding window chunking (chunk size and overlap configurable).

**Tips & Next Steps**
- For reproducible development, freeze Python to a narrower set of packages or use a smaller `requirements-dev.txt` for iteration.
- Consider adding a `.env` loader (python-dotenv) to keep secrets out of code.
- Add a minimal test or a short notebook demonstrating end-to-end ingest + query with sample data.

**Files of interest**
- `notebook/embed_sample.py` — produces embeddings and the parquet file used by `ingest.py`.
- `notebook/ingest.py` — shows how documents, embeddings, and metadata are stored in Chroma.
- `notebook/rag.py` — retrieval + prompt assembly for answer generation.
- `notebook/app.py` — minimal CLI wrapper to ask questions.

**License & Contact**
This repository is an educational demo. Contact the author/maintainer for licensing or reuse questions.
