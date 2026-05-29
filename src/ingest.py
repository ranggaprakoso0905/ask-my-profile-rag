import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.config import DOCS_DIR, QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

def load_markdown_documents():
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    return loader.load()

def load_pdf_documents():
    pdf_documents = []
    pdf_paths = list(Path(DOCS_DIR).glob("**/*.pdf"))
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = str(pdf_path)

        pdf_documents.extend(docs)

    return pdf_documents

def build_vectorstore(recreate_collection: bool = True):

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is not set.")
    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is not set.")

    markdown_docs = load_markdown_documents()
    pdf_docs = load_pdf_documents()

    documents = markdown_docs + pdf_docs

    print(f"Loaded {len(markdown_docs)} markdown documents.")
    print(f"Loaded {len(pdf_docs)} PDF pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
    )

    if recreate_collection:
        existing_collections = [
            collection.name
            for collection in client.get_collections().collections
        ]

        if QDRANT_COLLECTION_NAME in existing_collections:
            print(f"Deleting existing collection: {QDRANT_COLLECTION_NAME}")
            client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)       

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME
    )

    vectorstore.add_documents(chunks)

    print(f"Vectorstore saved to {QDRANT_COLLECTION_NAME}.")

if __name__ == "__main__":
    build_vectorstore()