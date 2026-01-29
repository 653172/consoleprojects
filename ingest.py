import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

UPLOAD_DIR = Path("uploads")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_pdf():
    UPLOAD_DIR.mkdir(exist_ok=True)

    pdfs = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDF found.")
        return None

    latest_pdf = max(pdfs, key=lambda p: p.stat().st_mtime)

    loader = PyPDFLoader(str(latest_pdf))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return db


if __name__ == "__main__":
    ingest_pdf()
