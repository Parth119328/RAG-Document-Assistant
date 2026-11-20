import streamlit as st
import random
import os
import tempfile
import hashlib

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📃",
    layout="wide"
)

st.title("📃 RAG Document Assistant")
st.caption(
    "Ask questions about your PDF and inspect the document chunks "
    "used to generate each answer."
)

st.subheader("📂 Upload your document")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    help="Upload the PDF you want to ask questions about."
)

if uploaded_file is None:
    st.info("Please upload a PDF to get started.")
    st.stop()

# Load the existing backend only once
@st.cache_resource(show_spinner=False)
def load_backend():
    import rag_engine
    import groq_implem

    return rag_engine, groq_implem


def process_uploaded_pdf(uploaded_file, rag_engine):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        extracted_text = rag_engine.loadPdf(temp_path)

        if not extracted_text.strip():
            raise ValueError("No readable text found in this PDF.")

        chunks = rag_engine.splitText(extracted_text)

        if not chunks:
            raise ValueError("Could not generate chunks from this PDF.")

        embeddings = rag_engine.generateEmbedding(chunks)

        return chunks, embeddings

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


with st.spinner("Loading the RAG engine and embedding model..."):
    try:
        rag_engine, groq_implem = load_backend()
    except Exception as error:
        st.error("Could not load the RAG backend.")
        st.exception(error)
        st.stop()

file_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()

if st.session_state.get("processed_file_id") != file_id:
    try:
        with st.spinner("Reading PDF and generating embeddings..."):
            chunks, embeddings = process_uploaded_pdf(uploaded_file, rag_engine)

        st.session_state["document_chunks"] = chunks
        st.session_state["document_embeddings"] = embeddings
        st.session_state["processed_file_id"] = file_id

        for key in ["last_question", "last_answer", "last_sources"]:
            st.session_state.pop(key, None)

    except Exception as error:
        st.error("Could not process the uploaded PDF.")
        st.exception(error)
        st.stop()

chunks = st.session_state["document_chunks"]
embeddings = st.session_state["document_embeddings"]

st.success(f"Document ready: {uploaded_file.name} | Total chunks: {len(chunks)}")

# Question input
def clear_question():
    st.session_state.question_input = ""

st.subheader("Ask your document")

random_placeholder = ["Need clarification on a topic? Ask here...", "What are the key takeaways from this document?", "Ask anything about your uploaded document...", "Looking for something specific? Ask away..."]
with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "Enter your question",
        placeholder=random.choice(random_placeholder),
        key="question_input"
    )

    submitted = st.form_submit_button(
        "Ask Question",
        type="primary",
        use_container_width=True
    )

# Retrieve context and generate the answer
if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                retrieved_chunks = rag_engine.retrieveChunks(
                    question,
                    chunks,
                    embeddings,
                    top_k=3
                )

                answer = groq_implem.generateAnswer(
                    question,
                    retrieved_chunks
                )

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = answer
            st.session_state["last_sources"] = retrieved_chunks

        except Exception as error:
            st.error("Something went wrong while answering your question.")
            st.exception(error)

# Display the generated answer
if "last_answer" in st.session_state:
    st.divider()

    st.subheader("💡 Generated Answer")
    st.write(st.session_state["last_answer"])

    st.caption(f"Question: {st.session_state['last_question']}")

st.divider()
st.caption("Built with Python, Streamlit, Sentence Transformers and Groq.")