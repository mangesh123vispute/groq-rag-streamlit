# Agentic RAG Project
<img width="1912" height="846" alt="image" src="https://github.com/user-attachments/assets/74e1332e-7d46-4531-998c-eabe6b66042d" />

A small **agentic Retrieval-Augmented Generation (RAG)** demo built with:

- **LangChain 0.3.x** + **LangGraph**
- **Groq** for the chat model (free tier at [console.groq.com](https://console.groq.com/))
- **FAISS** for vector search
- **Hugging Face** embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- **Streamlit** UI

The app loads **web URLs** and **local files** under `data/` (PDF, TXT, Markdown), builds a FAISS index, then runs a short graph: **retrieve → respond**. The LLM answers **only from retrieved passages** (your indexed documents), not from Wikipedia or other external sources.

In the **Streamlit** UI you can **upload** PDF / TXT / MD files; they are saved under **`data/uploads/`** and the index is **rebuilt** when you click **Save uploads & rebuild index**.

---

## Project structure

```text
RAG_PROJECT/
|-- main.py                    # CLI entry point
|-- streamlit_app.py           # Streamlit UI
|-- requirements.txt
|-- pyproject.toml
|-- uv.lock                    # if you use uv
|-- .streamlit/
|   `-- config.toml            # Streamlit server settings (e.g. file watcher)
|-- data/
|   |-- urls.txt              # optional: one URL per line (replaces default URLs when non-empty)
|   |-- uploads/              # user uploads from Streamlit (gitignored by default)
|   `-- *.pdf, *.txt, *.md   # anywhere under data/ (e.g. papers, notes)
`-- src/
    |-- config/config.py       # Groq model, chunking, default URLs, SSL workaround
    |-- document_ingestion/document_processor.py
    |-- vectorstore/vectorstore.py
    |-- graph_builder/graph_builder.py
    |-- node/reactnode.py      # generate_answer (context from index only)
    `-- state/rag_state.py
```

---

## How it works

1. **Ingestion** — `DocumentProcessor` loads each **HTTP(S) URL** in the active URL list, then **all** `.pdf`, `.txt`, and `.md` files **recursively** under `data/`. Files named `urls.txt` / `url.txt` are skipped (they are URL lists, not body text). Chunks are produced with `RecursiveCharacterTextSplitter` (default chunk size `500`, overlap `50`).

2. **Embeddings + index** — Chunks are embedded locally and stored in **FAISS**.

3. **Graph** — `GraphBuilder` defines a LangGraph flow: `retriever` → `responder` → end.

4. **Answering** — The responder prompts the LLM with retrieved passages only.

---

## Prerequisites

- **Python 3.12+** (`pyproject.toml` uses `requires-python = ">=3.12"`)
- A **Groq API key** (free)
- Internet access (URLs if you use them, Groq, Hugging Face model download on first run)

**Disk / RAM:** The first run downloads the sentence-transformer model and PyTorch-related wheels can be large. `torchvision` is included so Streamlit + `transformers` do not trip over optional vision imports.

---

## Installation

### Option A: `uv` (recommended)

```bash
uv sync
```

### Option B: `pip`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Use a **dedicated virtual environment** if you also use Anaconda globally—mixing installs often breaks LangChain versions.

---

## Environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a key at [https://console.groq.com/keys](https://console.groq.com/keys).

Do not commit real keys to git.

---

## Running the app

Run commands from the **project root** so paths like `data/` resolve correctly.

### CLI

```bash
python main.py
```

If **`data/urls.txt` does not exist**, **`Config.DEFAULT_URLS`** is used. If **`data/urls.txt` exists** (even when empty), **only the non-empty lines** in that file are used as URLs—an **empty file** means **no remote URLs** (local PDF/TXT/MD under `data/` only). **All** supported files under `data/` (including `data/uploads/`) are always loaded. The script prints example questions and can start an interactive loop.

### Streamlit

Prefer the same interpreter that has your dependencies:

```bash
python -m streamlit run streamlit_app.py
```

On Windows with a venv:

```text
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Then open the URL shown (usually `http://localhost:8501`).

### Uploading documents (Streamlit)

1. Open the **sidebar** → **Documents**.
2. Choose one or more **PDF**, **TXT**, or **Markdown** files.
3. Click **Save uploads & rebuild index**.

Files are written to **`data/uploads/`** with sanitized names. The app bumps an internal version, clears the chat history for that session, and **re-embeds** the full corpus (URLs + everything under `data/`). You can also drop files into `data/` manually and restart the app (or use the uploader once to trigger a rebuild if you add a dummy version bump—restarting Streamlit is simplest for manual drops).

User uploads are listed in [`.gitignore`](.gitignore) under `data/uploads/` so they are not committed by default.

---

## Streamlit configuration

[`.streamlit/config.toml`](.streamlit/config.toml) sets `fileWatcherType = "none"` so the dev server does not walk the entire `transformers` package tree (which can trigger noisy errors or missing optional deps). **Auto-reload on file save is off**; restart the app or use Streamlit’s rerun when you change code.

---

## Configuration

Main knobs in [`src/config/config.py`](src/config/config.py):

- `LLM_MODEL` — Groq model id (e.g. `llama-3.3-70b-versatile`; see [Groq docs](https://console.groq.com/docs/models))
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `DEFAULT_URLS` — used when `data/urls.txt` is missing or empty
- `DATA_DIR` / `UPLOADS_DIR` — project `data/` and `data/uploads/` paths

---

## Dependency notes (LangChain stack)

This repo pins a **0.3.x**-compatible stack so **Python 3.12** and **Groq** work together:

- `langchain-core` is capped below `0.3.86` (newer lines pull `langchain-protocol` patterns that expect **Python 3.13+** unless you upgrade everything).
- `langchain-groq` and `langchain-huggingface` stay on **pre-1.0** releases so they do not force `langchain-core` 1.x.

If you move to **Python 3.13 only** and want the newest LangChain 1.x partners, you can relax those pins deliberately—but treat it as a separate upgrade.

---

## Data sources

- **URLs** — If **`data/urls.txt` is missing**, use `Config.DEFAULT_URLS`. If **`data/urls.txt` exists**, use only the lines in it (empty file ⇒ no URLs; **CLI** and **Streamlit** behave the same).
- **Local files** — Any **`.pdf`**, **`.txt`**, or **`.md`** anywhere under **`data/`**, including **`data/uploads/`** from the UI. **`urls.txt` / `url.txt`** are not ingested as documents.

---

## Troubleshooting

- **`GROQ_API_KEY` / auth** — Set `.env` and restart the process.

- **`[Errno 2] No such file or directory` on startup (Windows + Anaconda)** — Often a broken **`SSL_CERT_FILE`** pointing at a missing `cacert.pem`. [`config.py`](src/config/config.py) clears invalid `SSL_CERT_FILE` on import; you can also unset it in your shell or fix the path in Anaconda.

- **`ImportError` / wrong LangChain** — Ensure Streamlit and `python` use the **same venv** (`python -m streamlit ...`). If imports still come from `anaconda3\Lib\site-packages`, your PATH is picking up the wrong environment.

- **Slow first run** — Embedding model download and FAISS build are one-time costs per fresh environment.

- **Weak answers** — Add more relevant documents, tune chunk size, or adjust retrieval `k` in code.

- **“No documents loaded”** — You need at least one URL that loads successfully and/or at least one PDF/TXT/MD under `data/`. For **offline / local-only**, add an **empty** `data/urls.txt` (or one with no non-empty lines) and put documents under `data/` or upload via Streamlit.

---

## License

Add your preferred license (MIT, Apache-2.0, etc.) here.
