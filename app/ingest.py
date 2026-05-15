import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path: str):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return chunks
