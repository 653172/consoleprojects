# 📄 Intelligent PDF Chatbot using RAG + OCR

An AI-powered document assistant that allows users to upload normal or scanned PDFs and ask natural language questions.  
Built using Retrieval Augmented Generation (RAG) to ensure responses are strictly grounded in the uploaded document.

---

## 🚀 Key Features

- Upload & replace PDFs dynamically
- Supports scanned/image-based PDFs using OCR
- Intelligent text chunking for better retrieval
- Semantic search with vector embeddings
- Context-aware AI responses
- No memory leakage between document uploads
- Fast local inference using Ollama (Llama3)

---

## 🏗 Architecture Overview

User → Streamlit UI → PDF Loader + OCR → Chunking → Embeddings → ChromaDB → Retriever → LLM → Response

---

## 🛠 Tech Stack

- Streamlit
- LangChain
- ChromaDB
- HuggingFace Sentence Transformers
- Tesseract OCR
- Ollama (Llama3)
note please delete the env folder and create your own virtual environment for better running 
---

## ▶️ Setup & Run

```bash
pip install -r requirements.txt
ollama run llama3
streamlit run app.py

