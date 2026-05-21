import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = "profile_docs"
VECTORSTORE_DIR = "vectorstore/faiss_index"

def build_vectorstore():

    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    Path(VECTORSTORE_DIR).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"Vectorstore saved to {VECTORSTORE_DIR}.")

if __name__ == "__main__":
    build_vectorstore()