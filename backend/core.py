import os
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain_pinecone import PineconeVectorStore
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Initialize Embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

#Initialize Vector Store
vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

#Initialize Chat Model
chat_model = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
)

@tool(response_format="content_and_artifact")
def retrieve_context(query:str):
    """Retrieve context from vector store based on query"""
    retrieved_docs = vectorstore.as_retriever().invoke(query,k=5)

    #serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('url','Unknown')}\n\nContent:{doc.page_content}" for doc in retrieved_docs)
    )
    return serialized, retrieved_docs


def run_llm(query:str) -> Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query (str): The user question
    
        Returns:
           Dictionary containing:
            - answeer : The generated answer
            - context: list of retreived documents
    """
    system_prompt = (
        """
        You are a helpful AI assistant that answers questions about Langchain Documentation.\n
        You have access to a tool that retrieves relevant documentation. \n
        Use the tool to find relevant information before answering questions. \n
        Always cite the sources you use in your answers. \n
        If you cannot find the answer in the retrieved documentation, say so.
    """
    )

    agent = create_agent(
        model="gpt-4o-mini",
        tools= [retrieve_context],
        system_prompt=system_prompt
    )
    messages = [HumanMessage(query)]
    response = agent.invoke({"messages": messages})

    answer = response["messages"][-1].content

    context_docs = []
    for message in response["messages"]:
        if isinstance(message,ToolMessage) and hasattr(message,"artifact"):
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    return {"answer":answer, "context":context_docs}

if __name__=="__main__":
    result = run_llm(query="What are deep agents?")
    print(result["answer"])