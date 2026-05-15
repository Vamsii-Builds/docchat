# 📄 DocChat — RAG-Powered PDF Chatbot

> Chat with your PDFs using hybrid retrieval (BM25 + vector search), FastAPI backend, and a clean Streamlit UI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

---

## 🧠 What It Does

DocChat lets you upload any PDF and ask natural language questions about it. It uses a **Retrieval-Augmented Generation (RAG)** architecture with hybrid search to find the most relevant chunks before generating an answer.

**Key design decisions explored:**
- Chunk size and overlap experiments (256 / 512 / 1024 tokens)
- Hybrid retrieval: BM25 (keyword) + FAISS (semantic) with reciprocal rank fusion
- Reranking with a cross-encoder to improve precision
- Dockerized for easy deployment

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
Text Extraction (PyMuPDF)
    │
    ▼
Chunking (RecursiveCharacterTextSplitter)
    │
    ├──► FAISS Vector Index (OpenAI Embeddings)
    └──► BM25 Sparse Index
             │
             ▼
         Hybrid Retriever (RRF Fusion)
             │
             ▼
         Reranker (cross-encoder)
             │
             ▼
         LLM (GPT-4o) → Answer
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.10+
- Docker (optional)
- OpenAI API key

### Local Setup

```bash
git clone https://github.com/vamsii-Builds/docchat
cd docchat
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=your_key_here

# Start the FastAPI backend
uvicorn app.main:app --reload

# In another terminal, start the UI
streamlit run ui/app.py
```

### Docker

```bash
docker build -t docchat .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key docchat
```

---

## 📁 Project Structure

```
docchat/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── ingest.py        # PDF parsing & chunking
│   ├── retriever.py     # Hybrid BM25 + FAISS retriever
│   └── chain.py         # LangChain RAG chain
├── ui/
│   └── app.py           # Streamlit frontend
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔬 Chunking Experiments

| Chunk Size | Overlap | Retrieval Precision (@5) | Notes |
|---|---|---|---|
| 256 | 30 | 0.61 | Too granular — loses context |
| 512 | 50 | 0.74 | Best balance |
| 1024 | 100 | 0.68 | Retrieves too much noise |

---

## 📌 What I Learned

- Hybrid retrieval consistently outperforms pure vector search on keyword-heavy queries
- Reranking adds ~8% precision improvement at the cost of ~40ms latency
- Chunk size matters more than model choice for many RAG use cases

---

## 🔗 Related Projects

- [FineTune Lab](https://github.com/vamsii-Builds/finetune-lab) — LLM fine-tuning with QLoRA
- [Job Tracker Agent](https://github.com/vamsii-Builds/job-tracker-agent) — Multi-step LangGraph agent
