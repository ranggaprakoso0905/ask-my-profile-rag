from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from src.retrieve import retrieve

import os

load_dotenv()

try:
    import streamlit as st

    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

except Exception:
    pass

# Set default settings
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

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

def generate_follow_up_questions(
        query: str,
        answer: str,
        context: str,
        llm_model: str = "claude-haiku-4-5-20251001",
):
    prompt = f"""
    You are helping generate follow-up questions for a RAG chatbot about Yoseph Widistika Rangga Prakoso's profile.

    Based on the current user question, assistant answer, and retrieved context, suggest 3 useful follow-up questions.

    Rules:
    - The questions should be short and natural.
    - The questions should help the user explore Yoseph's profile further.
    - Do not ask about information that is not supported by the context.
    - Return only the questions, one per line.
    - Do not use numbering or bullet points.

    User question:
    {query}

    Assistant answer:
    {answer}

    Retrieved context:
    {context}

    Follow-up questions:
    """

    llm = create_claude_llm(
        model=llm_model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=256
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    questions = [
        line.strip("-• 1234567890. ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return questions[:3]

def create_claude_llm(
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")
        
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
        
def answer_question(
        query: str,
        chat_history: list[dict] | None = None,
        llm_model: str = DEFAULT_LLM_MODEL
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
    
    llm = create_claude_llm(
        model=llm_model,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown source")
        if source not in sources:
            sources.append(source)

    follow_up_questions = generate_follow_up_questions(
    query=query,
    answer=response.content,
    context=context
)

    return {
        "answer": response.content,
        "sources": sources,
        "retrieved_docs": docs,
        "follow_up_questions": follow_up_questions
    }

if __name__ == "__main__":
    question = "What makes Yoseph suitable for a Data Scientist role?"
    result = answer_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print("-", source)