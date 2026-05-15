import streamlit as st
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
