import streamlit as st

from src.rag_chain import answer_question


st.set_page_config(
    page_title="Ask My Profile",
    page_icon="💬",
    layout="centered",
)

st.title("Ask My Profile")
st.caption("A RAG-powered chatbot for answering questions about Yoseph's profile.")

st.write(
    """
Ask questions about Yoseph's education, professional experience, projects, skills,
certifications, and career background.
"""
)


sample_questions = [
    "What makes Yoseph suitable for a Data Scientist role?",
    "What machine learning projects has Yoseph worked on?",
    "Does Yoseph have experience with Power BI?",
    "What is Yoseph's educational background?",
    "Does Yoseph have experience with computer vision?",
]


if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Sidebar for sample questions
with st.sidebar:
    st.header("About")
    st.write(
        """
        This app uses Retrieval-Augmented Generation (RAG) to answer questions about
        Yoseph's profile. It retrieves relevant information from a vector database and
        generates answers using a language model.
        """
    )


    st.header("Sample Questions")

    for sample_question in sample_questions:
        if st.button(sample_question):
            st.session_state.pending_question = sample_question
    

    st.header("Model Settings")

    selected_llm_model = st.selectbox(
        "LLM Model",
        [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1"
        ],
        index=0
    )

    st.header("Options")
    show_context = st.checkbox("Show retrieved context", value=False)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Always show the chat input
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

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Generate assistant answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating answer..."):
            try:
                result = answer_question(
                    question,
                    chat_history=st.session_state.messages,
                    llm_model=selected_llm_model
                )

                st.write(result["answer"])

                with st.expander("Sources"):
                    if result["sources"]:
                        for source in result["sources"]:
                            st.write(f"- `{source}`")
                    else:
                        st.write("No sources returned.")

                if show_context:
                    with st.expander("Retrieved context"):
                        for i, doc in enumerate(result["retrieved_docs"], start=1):
                            st.markdown(f"### Retrieved chunk {i}")
                            st.write(
                                f"Source: `{doc.metadata.get('source', 'unknown')}`"
                            )
                            st.write(doc.page_content)

                # Add assistant answer to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"]}
                )

            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)