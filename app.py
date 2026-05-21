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
    "What experience does Yoseph have in banking?",
    "What makes Yoseph suitable for a Data Analyst role?",
    "What machine learning projects has Yoseph worked on?",
    "Does Yoseph have experience with Power BI?",
    "What is Yoseph's educational background?",
    "Does Yoseph have experience with computer vision?",
]


with st.sidebar:
    st.header("About")
    st.write(
        """
This app retrieves relevant information from Yoseph's profile documents
and uses an LLM to generate an answer based on the retrieved context.
"""
    )

    st.header("Sample Questions")
    selected_question = st.selectbox(
        "Choose a sample question",
        [""] + sample_questions,
    )

    st.header("Settings")
    show_context = st.checkbox("Show retrieved context", value=False)

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# If user chooses a sample question, treat it as the next question
question = None

if selected_question:
    question = selected_question
else:
    question = st.chat_input("Ask a question about Yoseph's profile")


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