import os

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import VECTORSTORE_DIR, DOCS_DIR

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

def build_vectorstore():

    markdown_docs = load_markdown_documents()
    pdf_docs = load_pdf_documents()

    documents = markdown_docs + pdf_docs

    print(f"Loaded {len(markdown_docs)} markdown documents.")
    print(f"Loaded {len(pdf_docs)} PDF documents.")
    print(f"Loaded {len(documents)} total documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    Path(VECTORSTORE_DIR).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"Vectorstore saved to {VECTORSTORE_DIR}.")

if __name__ == "__main__":
    build_vectorstore()