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

def format_chat_history(chat_history: list[dict] | None, max_turns: int = 4):
    if not chat_history:
        return "No previous conversation."
    
    recent_history = chat_history[-max_turns * 2:]

    formatted_message = []

    for message in recent_history:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        formatted_message.append(f"{role.title()}: {content}")

    return "\n".join(formatted_message)
        
        
def answer_question(
        query: str,
        chat_history: list[dict] | None = None,
        llm_model: str = "gpt-5-mini"
    ):
    docs = retrieve(query, k=4)
    context = build_context(docs)
    history_text = format_chat_history(chat_history)

    prompt = f"""
You are an assistant answering questions about Yoseph Widistika Rangga Prakoso's profile.

Use the previous conversation only to understand follow-up questions.
Use the retrieved context as the source of factual information.
Do not make factual claims unless they are supported by the retrieved context.

Do not invent information or make assumptions. 
If the answer is not available in the context, say:
"I don't have enough information from the provided documents." 

Answer in a professional and informative style.
Use 2 to 4 short paragraphs to answer the question.
Include specific details from the context to support your answer.
Do not invent information not supported by the context.

Previous conversation:
{history_text}

Retrieved context:
{context}

Question:
{query}

Answer:
"""
    
    llm = ChatOpenAI(
        model=llm_model,
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