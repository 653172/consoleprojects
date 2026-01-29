import streamlit as st
import uuid
import tempfile
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document


# 🔧 Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- APP CONFIG ----------

st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.chat-box {
    padding: 1rem;
    border-radius: 12px;
    background: #ffffff;
    border: 1px solid #e6e6e6;
    margin-bottom: 1rem;
    color: #111;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}

.user {
    font-weight: 600;
    color: #2563eb;
}

.bot {
    color: #111;
}

.stTextInput > div > div > input {
    background: #1e1e1e;
    color: white;
    border-radius: 8px;
}

.stTextInput > div > div > input::placeholder {
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)


# ---------- SMART PDF LOADER ----------

def load_pdf_smart(pdf_path):
    docs = []

    # Try normal text
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except:
        docs = []

    # OCR fallback
    if not docs or sum(len(d.page_content.strip()) for d in docs) < 50:
        images = convert_from_path(pdf_path)

        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            docs.append(Document(page_content=text, metadata={"page": i + 1}))

    return docs


# ---------- VECTOR DB ----------

def build_fresh_db(pdf_path):
    docs = load_pdf_smart(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    temp_dir = Path(tempfile.gettempdir()) / f"rag_{uuid.uuid4().hex}"

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(temp_dir)
    )

    return db


# ---------- SIDEBAR ----------

with st.sidebar:
    st.markdown("### 📤 Upload Document")

    uploaded = st.file_uploader(
        "Upload or Replace PDF",
        type="pdf"
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded.read())
            pdf_path = f.name

        with st.spinner("Analyzing document..."):
            st.session_state.db = build_fresh_db(pdf_path)

        st.success("Document ready!")

    st.markdown("---")
    st.markdown("💡 Supports scanned & normal PDFs")


# ---------- MAIN ----------

st.markdown("## 📄 AI Document Assistant")
st.markdown("Ask questions and get precise answers from your PDF.")

if "db" not in st.session_state:
    st.info("Upload a PDF from the sidebar to start")

else:
    retriever = st.session_state.db.as_retriever(k=3)
    llm = ChatOllama(model="llama3")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.text_input("Type your question and press Enter")

    if question:
        with st.spinner("Searching document..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""
You are an AI assistant answering from a document.

Use ONLY the information in CONTENT to answer.
You may summarize or rephrase clearly.
If the answer cannot be found in CONTENT, say:
"Not found in document."

CONTENT:
{context}

QUESTION:
{question}

ANSWER:
"""


            answer = llm.invoke(prompt).content.strip()

        st.session_state.chat.append((question, answer))

    for q, a in reversed(st.session_state.chat):
        st.markdown(f"""
<div class="chat-box">
<span class="user">You:</span> {q}<br><br>
<span class="bot">AI:</span> {a}
</div>
""", unsafe_allow_html=True)
