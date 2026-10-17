
from langchain_ollama import ChatOllama
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableMap,
    RunnableParallel,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
import tempfile
load_dotenv()
embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0.2
)

def create_kb():
    """create a vector store from knowledge base documents and return the vector store"""   

    #split the KB into chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,    
        chunk_overlap=50,
    )
    doc = Document(
        page_content = KNOWLEDGE_BASE,
        metadata={"source": "langchain_knowledge_base.md"}
    )
    chunks = splitter.split_documents([doc])

    #create vector store from chunks
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings_model, persist_directory=tmpdir
        )
        print(
            f"Vector store created {vectorstore._collection.count()} documents and persisted."
        )
        return vectorstore

def demo_basic_rag():
    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        You are a question-answering assistant.

        Use ONLY the information provided in the context to answer the question.

        Context:
        {context}

        Question:
        {question}

        If the answer is present in the context, answer it directly.
        If the answer is not present in the context, say "I don't know."

        Answer:
        """
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    # Test
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    demo_basic_rag()