from langchain_community.vectorstores import FAISS
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
