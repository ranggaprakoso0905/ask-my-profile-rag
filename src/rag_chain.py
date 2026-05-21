from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.retrieve import retrieve

import os

load_dotenv()

try:
    import streamlit as st

    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

except Exception:
    pass

def build_context(docs):
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        content = doc.page_content

        context_parts.append(
            f"[Source {i}: {source}]\n{content}"
        )

    return "\n\n".join(context_parts)

def answer_question(query: str):
    docs = retrieve(query, k=4)
    context = build_context(docs)

    prompt = f"""
You are an assistant answering questions about Yoseph Widistika Rangga Prakoso's profile.

Use only the information in the context below.
Do not invent information or make assumptions. 
If the answer is not available in the context, say:
"I don't have enough information from the provided documents." 

Answer in a professional and informative style.
Use 2 to 4 short paragraphs to answer the question.
Include specific details from the context to support your answer.
Do not invent information not supported by the context.

Context:
{context}

Question:
{query}

Answer:
"""
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        if source not in sources:
            sources.append(source)

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_docs": docs
    }

if __name__ == "__main__":
    question = "What makes Yoseph suitable for a Data Scientist role?"
    result = answer_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)