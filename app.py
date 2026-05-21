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


if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if selected_question:
    st.session_state.question_input = selected_question


question = st.text_input(
    "Ask a question",
    key="question_input",
    placeholder="Example: What experience does Yoseph have in banking?",
)

ask_button = st.button("Ask", type="primary")

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            try:
                result = answer_question(question)

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                if result["sources"]:
                    for source in result["sources"]:
                        st.write(f"- `{source}`")
                else:
                    st.write("No sources returned.")

                if show_context:
                    st.subheader("Retrieved Context")
                    for i, doc in enumerate(result["retrieved_docs"], start=1):
                        with st.expander(f"Retrieved chunk {i}"):
                            st.write(f"Source: `{doc.metadata.get('source', 'unknown')}`")
                            st.write(doc.page_content)

            except Exception as e:
                st.error("An error occurred while generating the answer.")
                st.exception(e)