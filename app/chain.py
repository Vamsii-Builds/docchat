from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def build_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return chain
