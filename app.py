import streamlit as st

from src.rag_chain import answer_question

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []


st.set_page_config(
    page_title="ProfilePilot",
    page_icon="💬",
    layout="centered",
)

st.title("ProfilePilot")

st.write(
    """
Ask questions about Yoseph's education, professional experience, projects, skills,
certifications, and career background.
"""
)

sample_questions = [
    "What is Yoseph's background?",
    "What projects has Yoseph worked on?",
    "Please summarize Yoseph's professional experience.",
    "Does Yoseph have experience with computer vision?",
]

with st.sidebar:
    st.header("About")
    st.write(
        """
        This app uses Retrieval-Augmented Generation (RAG) to answer questions about
        Yoseph's profile. It retrieves relevant information from a vector database and
        generates answers using a language model.
        """
    )


# Show sample questions only before the conversation starts
if len(st.session_state.messages) == 0:
    st.subheader("Try asking:")

    cols = st.columns(2)

    for i, sample_question in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(sample_question, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending_question = sample_question
                st.rerun()

    st.divider()

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display suggested follow-up questions if available
if st.session_state.suggested_questions:
    st.write("Suggested follow-up questions:")

    for i, follow_up in enumerate(st.session_state.suggested_questions):
        if st.button(follow_up, key=f"follow_up_{i}", use_container_width=True):
            st.session_state.pending_question = follow_up
            st.session_state.suggested_questions = []
            st.rerun()

# Show clear button near the bottom after conversation starts
if len(st.session_state.messages) > 0:
    st.write("")

    if st.button("Clear conversation", use_container_width=True,type="primary"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.suggested_questions = []
        st.rerun()

# Always show chat input
typed_question = st.chat_input("Ask a question about Yoseph's profile")

question = None

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
elif typed_question:
    question = typed_question

if question:
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                result = answer_question(
                    question,
                    chat_history=st.session_state.messages,
                )

                st.write(result["answer"])

                # Add assistant answer to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )

                # Update suggested questions
                st.session_state.suggested_questions = result.get("follow_up_questions", [])

                # Rerun so the clear button and updated chat state render correctly
                st.rerun()

            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)