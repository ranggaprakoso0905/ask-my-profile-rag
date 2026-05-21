# Ask My Profile RAG

A Streamlit chatbot that answers questions about Yoseph Widistika Rangga Prakoso's profile using Retrieval-Augmented Generation (RAG).

The app retrieves relevant chunks from local Markdown profile documents, stores them in a FAISS vector index, and uses an OpenAI chat model to generate grounded answers with source references.

## Features

- Chat interface built with Streamlit
- Retrieval from Markdown profile documents in `profile_docs/`
- FAISS vector database for local semantic search
- Hugging Face sentence-transformer embeddings
- OpenAI-powered answer generation through LangChain
- Optional display of retrieved context for debugging and transparency
- Sample questions in the sidebar

## Project Structure

```text
.
|-- app.py                         # Streamlit app entry point
|-- requirements.txt               # Python dependencies
|-- profile_docs/                  # Source Markdown documents for the profile
|-- src/
|   |-- config.py                  # App configuration
|   |-- ingest.py                  # Builds the FAISS vectorstore
|   |-- retrieve.py                # Loads/searches the vectorstore
|   `-- rag_chain.py               # RAG prompt and OpenAI call
|-- vectorstore/                   # Generated FAISS index, ignored by git
|-- .streamlit/config.toml         # Streamlit config
`-- .devcontainer/devcontainer.json
```

## Requirements

- Python 3.11 recommended
- OpenAI API key
- Internet access when installing dependencies and downloading the embedding model

## Setup

Create and activate a virtual environment:

```powershell
python -m venv rag-env
.\rag-env\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Optional: override the default embedding model:

```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## Build the Vectorstore

The app can build the vectorstore automatically the first time it needs retrieval. You can also build it manually:

```powershell
python -m src.ingest
```

This reads Markdown files from `profile_docs/`, splits them into chunks, embeds them, and saves the FAISS index to `vectorstore/faiss_index`.

## Run the App

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Updating Profile Content

Edit or add Markdown files under `profile_docs/`, then rebuild the vectorstore:

```powershell
python -m src.ingest
```

If `vectorstore/` already exists, rebuilding refreshes the generated index from the current documents.

## Configuration

The main configuration values are in `src/config.py`:

- `DOCS_DIR`: source document folder, default `profile_docs`
- `VECTORSTORE_DIR`: FAISS index path, default `vectorstore/faiss_index`
- `EMBEDDING_MODEL_NAME`: embedding model, default `sentence-transformers/all-MiniLM-L6-v2`

The OpenAI chat model is configured in `src/rag_chain.py` and currently uses `gpt-4o-mini` with temperature `0`.

## Notes

- `.env`, `vectorstore/`, local virtual environments, and local model files are ignored by git.
- For Streamlit Community Cloud or similar deployments, provide `OPENAI_API_KEY` through Streamlit secrets.
- Retrieved source files are shown below each answer in the app.
