import os
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME
from src.ingest import build_vectorstore

def load_vectorstore():

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set.")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is not set.")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
        )

    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
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
        