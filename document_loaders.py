import os
import pathlib as Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
DirectoryLoader, 
PyPDFLoader,
TextLoader, 
CSVLoader, 
PDFMinerLoader, 
UnstructuredWordDocumentLoader, 
UnstructuredPowerPointLoader, 
UnstructuredMarkdownLoader   )

def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} documents from {pdf_path}")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}:")
        print(f"Page Content: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    # Example usage
    pdf_path = r"C:\Users\ADMIN\OneDrive\Desktop\DS & ML\RAG\Langc_course\docs\langchain_demo.pdf"  # Replace with your PDF file path
    pdf_loader(pdf_path)