import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import EMBEDDING_MODEL_NAME, VECTORSTORE_DIR
from src.ingest import build_vectorstore

def load_vectorstore():
    if not Path(VECTORSTORE_DIR).exists():
        build_vectorstore()

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vectorstore

def retrieve(query: str, k: int = 4):
    vectorstore = load_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return results

if __name__ == "__main__":
    query = "What experience does Yoseph have in banking?"
    results = retrieve(query)

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}:")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:800])
        