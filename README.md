# Ask My Profile RAG

ProfilePilot is a Streamlit chatbot that answers questions about Yoseph Widistika Rangga Prakoso's profile using Retrieval-Augmented Generation (RAG).

The app loads local Markdown and PDF profile documents, chunks them with LangChain, stores embeddings in a Qdrant collection, retrieves relevant context with MMR search, and uses Anthropic Claude to generate grounded answers and suggested follow-up questions.

## Features

- Streamlit chat interface with sample questions and conversation reset
- Retrieval from Markdown and PDF documents in `profile_docs/`
- Qdrant vector database for semantic search
- OpenAI `text-embedding-3-small` embeddings
- Anthropic Claude answer generation through LangChain
- Claude-generated suggested follow-up questions
- Streamlit secrets and local `.env` configuration support

## Current Stack

- Python
- Streamlit
- LangChain
- Qdrant / `qdrant-client`
- OpenAI embeddings via `langchain-openai`
- Anthropic chat models via `langchain-anthropic`
- `pypdf` for PDF document loading
- `python-dotenv` for local environment variables

## Project Structure

```text
.
|-- app.py                         # Streamlit app entry point
|-- requirements.txt               # Python dependencies
|-- profile_docs/                  # Local source documents, ignored by git
|-- src/
|   |-- config.py                  # Environment and Streamlit secret loading
|   |-- ingest.py                  # Loads docs and builds the Qdrant collection
|   |-- retrieve.py                # Loads Qdrant and retrieves relevant chunks
|   `-- rag_chain.py               # RAG prompts, Claude calls, and follow-ups
|-- previous_ver/                  # Older implementation experiments, ignored by git
|-- .streamlit/config.toml         # Streamlit config
`-- .devcontainer/devcontainer.json
```

## Requirements

- Python 3.11+ recommended
- OpenAI API key for embeddings
- Anthropic API key for answer generation
- Qdrant URL and API key
- Internet access for installing dependencies and calling model/vector APIs

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
ANTHROPIC_API_KEY=your_anthropic_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=ask_my_profile
```

`QDRANT_COLLECTION_NAME` is optional. It defaults to `ask_my_profile`.

## Build the Vector Store

Build or refresh the Qdrant collection:

```powershell
python -m src.ingest
```

This command reads Markdown and PDF files from `profile_docs/`, splits them into chunks, embeds them with OpenAI embeddings, and writes them to the configured Qdrant collection.

By default, `src.ingest.build_vectorstore()` recreates the collection if it already exists.

## Run the App

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Updating Profile Content

Edit or add files under `profile_docs/`, then rebuild the Qdrant collection:

```powershell
python -m src.ingest
```

The current loader supports:

- Markdown files: `**/*.md`
- PDF files: `**/*.pdf`

## Configuration

The main configuration values live in `src/config.py`:

- `DOCS_DIR`: source document folder, default `profile_docs`
- `QDRANT_URL`: Qdrant cluster URL
- `QDRANT_API_KEY`: Qdrant API key
- `QDRANT_COLLECTION_NAME`: Qdrant collection name, default `ask_my_profile`
- `OPENAI_API_KEY`: used by `OpenAIEmbeddings`
- `ANTHROPIC_API_KEY`: used by Claude chat models

Model settings are currently defined in code:

- Embeddings: `text-embedding-3-small` in `src/ingest.py` and `src/retrieve.py`
- Main answer model: `claude-sonnet-4-6` in `src/rag_chain.py`
- Follow-up model: `claude-haiku-4-5-20251001` in `src/rag_chain.py`

## Notes

- `.env`, `profile_docs/`, local virtual environments, local model files, and `previous_ver/` are ignored by git.
- For Streamlit Community Cloud or similar deployments, provide API keys and Qdrant settings through Streamlit secrets.
- The app currently retrieves sources internally, but the UI only displays the generated answer and suggested follow-up questions.
