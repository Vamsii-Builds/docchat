files = {
'app/ingest.py': '''import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks
''',
'app/retriever.py': '''from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

def build_retriever(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 4
    hybrid = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )
    return hybrid
''',
'app/chain.py': '''from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain
''',
'ui/app.py': '''import streamlit as st
import tempfile, os
from dotenv import load_dotenv
from app.ingest import load_and_chunk
from app.retriever import build_retriever
from app.chain import build_chain

load_dotenv()

st.title("DocChat - Chat with your PDF")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        tmp_path = f.name
    with st.spinner("Processing PDF..."):
        chunks = load_and_chunk(tmp_path)
        retriever = build_retriever(chunks)
        chain = build_chain(retriever)
    st.success(f"Ready! {len(chunks)} chunks indexed.")
    question = st.text_input("Ask a question about your PDF:")
    if question:
        with st.spinner("Thinking..."):
            result = chain.invoke({"query": question})
            st.write("**Answer:**", result["result"])
    os.unlink(tmp_path)
'''
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created {path}")

print("All files created!")