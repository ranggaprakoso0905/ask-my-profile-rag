import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from src.config import ANTHROPIC_API_KEY
from src.retrieve import retrieve

# Set default settings
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024

# Function to build context from retrieved documents
def build_context(docs):
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        content = doc.page_content

        context_parts.append(
            f"[Source {i}: {source}]\n{content}"
        )

    return "\n\n".join(context_parts)

# Function to format chat history for the prompt
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

# Function to create a Claude LLM instance
def create_claude_llm(
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS
):
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")
        
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

# Function to generate follow-up questions
def generate_follow_up_questions(
        query: str,
        answer: str,
        context: str,
        llm_model: str = "claude-haiku-4-5-20251001",
):
    prompt = f"""
    You are helping generate follow-up questions for a RAG chatbot about Yoseph Widistika Rangga Prakoso's profile.

    Based on the current user question, assistant answer, and retrieved context, suggest 3 useful follow-up questions.

    Strict rules:
    - Each question must be under 12 words.
    - The questions should help the user explore Yoseph's profile further.
    - Do not ask about information that is not supported by the context.
    - Focus on asking about Yoseph's skills, experience, projects, or background related to data science.
    - Do not create speculative links between unrelated topics.
    - Do not ask questions that assume a relationship unless the relationship is explicitly stated in the context.
    - Do not combine unrelated areas such as civil engineering and machine learning unless the context clearly connects them.
    - Return only the questions, one per line.
    - Do not use numbering or bullet points.

    Good examples:
    What projects has Yoseph worked on?
    What data science skills does Yoseph have?
    What are Yoseph's current career aspirations?

    Bad examples:
    How does Yoseph's civil engineering background relate to machine learning?
    How did his scholarship shape his computer vision work?
    How does banking directly connect to his computer vision work?

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
        max_tokens=128
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    questions = [
        line.strip("-• 1234567890. ").strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return questions[:3]

# Main function to answer a user question
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

    Your task:
    Answer the user's question using the retrieved context below.

    Grounding rules:
    - Use the retrieved context as the only source of factual information.
    - Use the previous conversation only to understand follow-up questions.
    - Do not invent information, infer unsupported facts, or make assumptions.
    - If the retrieved context does not contain enough information, say exactly:
      "I don't have enough information from the provided documents."

    Formatting rules:
    - Do not start with a title or Markdown heading.
    - You can use bold text for emphasis.
    - Use 2 to 4 short paragraphs when useful.
    - Use bullet points only if the user asks for a list or if it improves clarity.
    - Keep the tone professional, informative, and natural.

    Content rules:
    - Include specific details from the retrieved context when available.
    - If the question asks for a summary, synthesize the most relevant points.
    - If the question is a follow-up, connect it to the previous conversation, but still ground the answer in the retrieved context.

    Previous conversation:
    {history_text}

    Retrieved context:
    {context}

    User question:
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
    context=context[:2000]
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