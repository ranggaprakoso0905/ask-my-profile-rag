import os

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_DIR = "vectorstore/faiss_index"
DOCS_DIR = "profile_docs"