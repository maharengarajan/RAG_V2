import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_TOKEN=os.getenv("HUGGINGFACE_TOKEN")


CHROMA_PATH="Chroma"
DATA_PATH="data"


def load_documents():
    document_loader=PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    db=Chroma.from_documents(documents=chunks,embedding=get_embedding_function(),persist_directory=CHROMA_PATH)
    return db


if __name__=="__main__":
    documents=load_documents()
    chunks=split_documents(documents=documents)
    add_to_chroma(chunks=chunks)