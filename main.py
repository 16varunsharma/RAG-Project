from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
#rom langgraph import __version__ as lg_version
import importlib.metadata
import importlib.metadata
lg_version = importlib.metadata.version("langgraph")
from langchain_openai import ChatOpenAI

from langchain_ollama import ChatOllama 

print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}") 
#rint(f"langchain_openai Version: {ChatOpenAI.__version__}")
print(f"langchain_openai Version: {importlib.metadata.version('langchain-openai')}")

def main():
    #test openai 
    print("Hello from langc-course!")
    llm = ChatOllama(model="llama3.2", temperature=0)
    response = llm.invoke("Say 'setup complete: in one word")
    print(f"Response from LLM: {response}")


    print("Setup complete")
if __name__ == "__main__":
    main()

