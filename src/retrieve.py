import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR
from src.ingest import build_vectorstore

def load_vectorstore():
    if not Path(VECTORSTORE_DIR).exists():
        build_vectorstore()

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
        )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return vectorstore

def retrieve(query: str, k: int = 6):
    vectorstore = load_vectorstore()
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 12}
    )

    results = retriever.invoke(query)

    return results

if __name__ == "__main__":
    query = "What experience does Yoseph have in banking?"
    results = retrieve(query)

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}:")
        print("Source:", doc.metadata.get("source"))
        print(doc.page_content[:800])
        